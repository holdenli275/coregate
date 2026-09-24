"""Graph-transformer encoder for CoreGate.

Two-layer, 128-channel message passing over the *full* telemetry graph: no node
is removed before or during propagation, which is what lets the released runs
compare supervision sources without changing the representation.

Inputs follow the paper.  Node features combine 12 log-transformed count
channels with four entity-type indicators.  Edge features encode the operation
in eight bins, and the order encoder adds a sinusoidal embedding of the
normalized edge order ``te = (e - 1) / max(1, |E| - 1)``.  The released graphs
store edges as an ordered list, so that coordinate is available without
physical timestamps; the low/high end of the order still distinguishes early
from late interactions, which the pooling in :mod:`coregate.tgn_episode` reads.

The encoder never reads the execution journal or a target mask.  It is shared by
every matched model in the paper (Ordinary, FullRule, CoreGate-Lineage, the
SAGPool variant, and the shuffled-target control), and the foreground
supervision enters only through the node gate and the loss.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class OrderEncoder(nn.Module):
    """Sinusoidal embedding of the normalized edge order in ``[0, 1]``."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, order):
        """
        Args:
            order: [num_edges] normalized edge order
        Returns:
            order_embeds: [num_edges, dim]
        """
        device = order.device
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device).float() *
            (-np.log(10000.0) / self.dim)
        )

        pe = torch.zeros(order.size(0), self.dim, device=device)
        pe[:, 0::2] = torch.sin(order.unsqueeze(-1) * div_term)
        pe[:, 1::2] = torch.cos(order.unsqueeze(-1) * div_term)

        return pe


class GraphTransformerEncoder(nn.Module):
    """Full-graph encoder with a node-level technique head.

    ``num_subtechs`` is the number of labeled classes in the catalog (45 in the
    released study).  The sub-technique head is the node classifier whose
    graph-mean logits enter the classification loss; the node states it is
    computed from are also what the readout in :mod:`coregate.tgn_episode`
    pools.
    """

    def __init__(self, node_feat_dim=16, edge_feat_dim=8, hidden_dim=128,
                 num_layers=2, num_tactics=1, num_techniques=1,
                 num_subtechs=45, num_classes=None, dropout=0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Kept for callers that pass a single class count.
        if num_classes is not None:
            num_subtechs = num_classes

        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.node_bn = nn.BatchNorm1d(hidden_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

        self.order_encoder = OrderEncoder(hidden_dim)

        self.gat_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // 4,
                heads=4,
                dropout=0.1,
                concat=True,
                edge_dim=hidden_dim
            )
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.layer_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # The node classifier: graph-mean logits over its output feed the
        # classification loss.
        self.subtech_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_subtechs)
        )

        # Auxiliary heads.  Every released run builds them with a single output
        # each, so they carry no class information, but they are part of the
        # 225,839-parameter Ordinary model reported in the paper, and keeping
        # them reproduces the released configuration exactly.
        self.tactic_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tactics)
        )

        self.technique_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_techniques)
        )

    def forward(self, x, edge_index, edge_attr, edge_time, batch=None):
        """
        Args:
            x: [num_nodes, node_feat_dim] node features
            edge_index: [2, num_edges] edge index
            edge_attr: [num_edges, edge_feat_dim] edge features
            edge_time: [num_edges] normalized edge order
            batch: unused, kept for interface compatibility
        Returns:
            node_logits: [num_nodes, num_subtechs] per-node technique logits
            node_states: [num_nodes, hidden_dim] node representations
        """
        x = self.node_proj(x)
        if x.size(0) > 1:
            x = self.node_bn(x)
        x = F.relu(x)

        if edge_attr is not None and len(edge_attr) > 0 and edge_index.size(1) > 0:
            edge_attr = self.edge_proj(edge_attr)
            if edge_time is not None and len(edge_time) > 0:
                order_embed = self.order_encoder(edge_time)
                edge_attr = edge_attr + order_embed
        else:
            if edge_index.size(1) == 0:
                num_nodes = x.size(0)
                edge_index = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
                edge_attr = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
            else:
                edge_attr = None

        for gat_layer, norm, bn in zip(self.gat_layers, self.layer_norms, self.layer_bns):
            x_new = gat_layer(x, edge_index, edge_attr=edge_attr)
            x_new = norm(x_new)
            if x_new.size(0) > 1:
                x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new

        node_logits = self.subtech_classifier(x)

        return node_logits, x
