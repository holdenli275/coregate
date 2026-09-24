#!/usr/bin/env python3
"""Shared graph construction and model helpers for the CoreGate runners."""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from coregate.graph_encoder import GraphTransformerEncoder
from linux_graph_loader import graph_tensor


def load_rows(catalog, graph_field):
    by_class = defaultdict(list)
    for row in catalog["samples"]:
        if row.get("target"):
            by_class[row["target"]].append(row)
    rows = []
    for label in sorted(by_class):
        values = sorted(by_class[label], key=lambda row: row["sample_id"])
        test_count = 1 if len(values) == 2 else min(2, len(values) - 1)
        rows.extend((row, label, index >= len(values) - test_count)
                     for index, row in enumerate(values))
    graphs = [graph_tensor(row[graph_field]) for row, _, _ in rows]
    labels = np.asarray([label for _, label, _ in rows])
    test_mask = np.asarray([is_test for _, _, is_test in rows])
    return graphs, labels, test_mask, sorted(by_class)


def encoder_input(graph, class_id):
    """Project a loaded Linux graph tensor into the encoder's 16/8 interface."""
    counts = torch.log1p(graph["x"][:, :6])
    reverse_counts = torch.log1p(graph["x"][:, 21:27])
    node_types = F.one_hot(graph["node_type"].long(), num_classes=4).float()
    x = torch.cat([counts, reverse_counts, node_types], dim=1)
    edge_type = graph["edge_type"].long()
    edge_attr = F.one_hot(edge_type.remainder(8), num_classes=8).float()
    edge_count = graph["edge_index"].shape[1]
    # Normalized edge order, not a physical timestamp: the released graphs store
    # edges as an ordered list.  See the paper's edge-order coordinate.
    edge_time = (torch.arange(edge_count, dtype=torch.float32) /
                 max(1, edge_count - 1))
    return Data(
        x=x, edge_index=graph["edge_index"], edge_attr=edge_attr,
        edge_time=edge_time, y=torch.tensor(class_id, dtype=torch.long),
        node_type=graph.get("node_type", torch.zeros(x.size(0), dtype=torch.long)),
        core_node_mask=graph.get("core_node_mask", torch.zeros(
            x.size(0), dtype=torch.bool)),
        core_edge_mask=graph.get("core_edge_mask", torch.zeros(
            edge_count, dtype=torch.bool)),
    )
