"""Episode-level training and pooling for the CoreGate encoder.

The module name keeps the internal ``episode_tgn`` protocol token that every
released run JSON records as ``training_protocol``; the pipeline itself is the
static one described in the paper (graph-transformer encoder, gated readout).

Training uses one fixed support graph per class.  The evidence gate is learned
only from source-graph targets; at inference the gate is computed from node
representations, and neither the journal nor a stored mask is read.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGPooling


def temporal_graph_forward(model, data, core_pool_weight=0.0,
                           pooling="ordinary", return_node_embeddings=False):
    """Return node logits and a graph embedding with edge-time pooling."""
    node_logits, node_embeddings = model(
        data.x, data.edge_index, data.edge_attr, data.edge_time
    )
    node_count = node_embeddings.size(0)
    if data.edge_index.numel() == 0 or node_count == 0:
        result = (node_logits.mean(dim=0), node_embeddings.mean(dim=0))
        return (*result, node_embeddings) if return_node_embeddings else result
    # Each node receives the mean edge order of its incident edges.  The encoder
    # already adds a sinusoidal embedding of the edge order to every edge
    # feature; this pool keeps the early/late distinction at graph level instead
    # of averaging it away completely.
    node_time_sum = torch.zeros(node_count, device=node_embeddings.device)
    node_degree = torch.zeros(node_count, device=node_embeddings.device)
    for endpoint in (data.edge_index[0], data.edge_index[1]):
        node_time_sum.index_add_(0, endpoint, data.edge_time)
        node_degree.index_add_(0, endpoint, torch.ones_like(data.edge_time))
    node_time = node_time_sum / node_degree.clamp_min(1.0)
    weights = torch.softmax(node_time, dim=0)
    if pooling == "sagpool":
        sag_pool = getattr(model, "sag_pool", None)
        if sag_pool is None:
            raise RuntimeError("SAGPool readout was not initialized")
        batch = torch.zeros(node_count, dtype=torch.long,
                            device=node_embeddings.device)
        pooled_nodes, _, _, pooled_batch, permutation, _ = sag_pool(
            node_embeddings, data.edge_index, batch=batch)
        if pooled_nodes.size(0) == 0:
            pooled = node_embeddings.mean(dim=0)
        else:
            selected_time = node_time[permutation]
            selected_weights = torch.softmax(selected_time, dim=0)
            pooled = 0.5 * pooled_nodes.mean(dim=0) + 0.5 * (
                pooled_nodes * selected_weights.unsqueeze(1)).sum(dim=0)
        result = (node_logits.mean(dim=0), pooled)
        return (*result, node_embeddings) if return_node_embeddings else result
    if pooling in {"norm_attention", "logit_attention", "predicted_core",
                   "learned_core"}:
        # Deployable alternative: infer graph attention from the learned node
        # representation itself, with no Core_Graph or attack mask at test
        # time.  Temperature 1 keeps the attention conservative.
        if pooling == "learned_core":
            gate = getattr(model, "core_gate", None)
            if gate is None:
                scores = torch.linalg.vector_norm(node_embeddings, dim=1)
            else:
                scores = gate(node_embeddings).squeeze(-1)
        elif pooling == "predicted_core":
            scores = getattr(data, "core_score", None)
            if scores is None or scores.numel() != node_count:
                scores = torch.zeros(node_count, device=node_embeddings.device)
            scores = scores.to(node_embeddings.device).float()
        elif pooling == "logit_attention":
            # The node-level technique head is trained by the graph label
            # objective.  Its maximum class confidence is a weak, deployable
            # attack-evidence score and does not require a Core_Graph mask.
            scores = torch.softmax(node_logits, dim=1).max(dim=1).values
        else:
            scores = torch.linalg.vector_norm(node_embeddings, dim=1)
        attention = torch.softmax(scores, dim=0)
        pooled = 0.5 * node_embeddings.mean(dim=0) + 0.5 * (
            node_embeddings * attention.unsqueeze(1)).sum(dim=0)
        result = (node_logits.mean(dim=0), pooled)
        return (*result, node_embeddings) if return_node_embeddings else result
    if pooling == "core" and core_pool_weight > 0:
        core = getattr(data, "core_node_mask", None)
        if core is not None and core.numel() == node_count and bool(core.any()):
            # The evidence core is often tiny relative to host background.
            # Reweighting it at pooling time keeps the classifier from being
            # dominated by environment-specific peripheral volume.
            boost = torch.ones(node_count, device=node_embeddings.device)
            boost[core] = float(core_pool_weight)
            weights = weights * boost
            weights = weights / weights.sum().clamp_min(1e-8)
            mean_weights = boost / boost.sum().clamp_min(1e-8)
            core_mean = (node_embeddings * mean_weights.unsqueeze(1)).sum(dim=0)
            weighted = (node_embeddings * weights.unsqueeze(1)).sum(dim=0)
            pooled = 0.5 * core_mean + 0.5 * weighted
            result = (node_logits.mean(dim=0), pooled)
            return (*result, node_embeddings) if return_node_embeddings else result
    weighted = (node_embeddings * weights.unsqueeze(1)).sum(dim=0)
    pooled = 0.5 * node_embeddings.mean(dim=0) + 0.5 * weighted
    result = (node_logits.mean(dim=0), pooled)
    return (*result, node_embeddings) if return_node_embeddings else result


def episode_split(labels, support_per_class=1):
    """Return deterministic support/query indices for a training episode."""
    support = {}
    query = []
    for index, label in enumerate(labels):
        values = support.setdefault(int(label), [])
        if len(values) < support_per_class:
            values.append(index)
        else:
            query.append(index)
    return support, query


def _prototype_loss(embedding, label, prototypes, prototype_labels,
                    temperature=0.20, margin=0.20):
    embedding = F.normalize(embedding.unsqueeze(0), dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    similarities = embedding @ prototypes.t()
    target_index = prototype_labels.index(int(label))
    target = torch.tensor([target_index], dtype=torch.long, device=embedding.device)
    ce = F.cross_entropy(similarities / temperature, target)
    positive = similarities[0, target_index]
    negatives = torch.cat([similarities[0, :target_index],
                           similarities[0, target_index + 1:]])
    hard_negative = negatives.max() if len(negatives) else positive.detach()
    margin_loss = F.relu(margin - positive + hard_negative)
    return ce, margin_loss


@torch.no_grad()
def episode_prototypes(model, data_list, labels, support, core_pool_weight=0.0,
                       pooling="ordinary"):
    model.eval()
    embeddings = {}
    for label, indices in support.items():
        embeddings[label] = torch.stack([
            temporal_graph_forward(model, data_list[index], core_pool_weight,
                                   pooling)[1]
            for index in indices
        ]).mean(dim=0)
    classes = sorted(embeddings)
    return classes, torch.stack([embeddings[label] for label in classes])


def train_episode_tgn(model, data_list, labels, epochs, seed,
                      support_per_class=1, metric_weight=0.50,
                      margin_weight=0.10, temperature=0.20, margin=0.20,
                      core_pool_weight=0.0,
                      pooling="ordinary",
                      core_gate_weight=0.0,
                      share_gate_forward=False):
    """Train with one support/query episode per epoch over the train split."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    labels = [int(label) for label in labels]
    support, query = episode_split(labels, support_per_class)
    # The evidence gate is learned from training Core_Graph masks only.  At
    # inference, ``learned_core`` predicts the gate from node embeddings and
    # never reads the held-out graph's mask.
    if pooling == "learned_core" and not hasattr(model, "core_gate"):
        model.core_gate = nn.Linear(model.hidden_dim, 1)
    if pooling == "sagpool" and not hasattr(model, "sag_pool"):
        model.sag_pool = SAGPooling(model.hidden_dim, ratio=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    last_loss = None
    last_parts = {}
    for _ in range(epochs):
        prototype_labels, prototypes = episode_prototypes(
            model, data_list, labels, support, core_pool_weight, pooling)
        order = list(range(len(data_list)))
        random.shuffle(order)
        model.train()
        for index in order:
            forward_result = temporal_graph_forward(
                model, data_list[index], core_pool_weight, pooling,
                return_node_embeddings=share_gate_forward)
            if share_gate_forward:
                logits, embedding, shared_node_embeddings = forward_result
            else:
                logits, embedding = forward_result
            target = torch.tensor([labels[index]], dtype=torch.long)
            ce = F.cross_entropy(logits.unsqueeze(0), target)
            if index in query:
                metric_ce, margin_loss = _prototype_loss(
                    embedding, labels[index], prototypes, prototype_labels,
                    temperature=temperature, margin=margin,
                )
            else:
                metric_ce = torch.zeros((), device=embedding.device)
                margin_loss = torch.zeros((), device=embedding.device)
            loss = ce + metric_weight * metric_ce + margin_weight * margin_loss
            gate_loss = torch.zeros((), device=embedding.device)
            if pooling == "learned_core" and core_gate_weight > 0:
                core = getattr(data_list[index], "core_node_mask", None)
                gate = getattr(model, "core_gate", None)
                if gate is not None and core is not None and core.numel() == data_list[index].x.size(0):
                    if share_gate_forward:
                        gate_logits = gate(shared_node_embeddings).squeeze(-1)
                    else:
                        gate_logits = gate(model(data_list[index].x,
                                                 data_list[index].edge_index,
                                                 data_list[index].edge_attr,
                                                 data_list[index].edge_time)[1]).squeeze(-1)
                    target_core = core.to(gate_logits.device).float()
                    positives = target_core.sum()
                    negatives = target_core.numel() - positives
                    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
                    gate_loss = F.binary_cross_entropy_with_logits(
                        gate_logits, target_core, pos_weight=pos_weight)
                    loss = loss + core_gate_weight * gate_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.detach())
            last_parts = {"ce": float(ce.detach()),
                          "metric_ce": float(metric_ce.detach()),
                          "margin": float(margin_loss.detach()),
                          "core_gate": float(gate_loss.detach())}
    return last_loss, last_parts, support, query


@torch.no_grad()
def embed_episode_graphs(model, data_list, core_pool_weight=0.0,
                         pooling="ordinary"):
    model.eval()
    return torch.stack([
        temporal_graph_forward(model, data, core_pool_weight, pooling)[1]
        for data in data_list
    ])
