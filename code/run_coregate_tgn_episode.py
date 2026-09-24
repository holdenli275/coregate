#!/usr/bin/env python3
"""Episode-level support/query training for the CoreGate graph encoder."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import graph_base as base
from coregate.tgn_episode import (
    embed_episode_graphs,
    episode_split,
    train_episode_tgn,
)
from coregate.graph_cache import load_cached_graphs


def metrics(prediction, labels):
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def support_similarity(train_embeddings, test_embeddings, labels, support):
    support_labels = sorted(support)
    support_embeddings = torch.stack([
        train_embeddings[support[label]].mean(dim=0) for label in support_labels
    ])
    support_embeddings = F.normalize(support_embeddings, dim=1)
    train_normalized = F.normalize(train_embeddings, dim=1)
    test_normalized = F.normalize(test_embeddings, dim=1)
    return (train_normalized @ support_embeddings.t()).numpy(), \
        (test_normalized @ support_embeddings.t()).numpy()


def all_class_support(labels):
    """Use every training example as a class prototype support set."""
    values = np.asarray(labels)
    return {int(label): np.flatnonzero(values == label).tolist()
            for label in sorted(set(values.tolist()))}


def new_model(classes):
    return base.GraphTransformerEncoder(
        node_feat_dim=16, edge_feat_dim=8, hidden_dim=128, num_layers=2,
        num_tactics=1, num_techniques=1, num_subtechs=classes, dropout=0.2,
    )


def make_classifier(kind, seed, c):
    if kind == "svm":
        return SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced",
                   probability=True, random_state=seed)
    if kind == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return LogisticRegression(max_iter=2000, class_weight="balanced",
                              random_state=seed, C=c)


def load_rows_with_environment(catalog, graph_field):
    """Match ``base.load_rows`` while retaining the catalog environment.

    Keeping the split construction byte-for-byte equivalent makes the
    invariant and ordinary episode results directly comparable.
    """
    from collections import defaultdict

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
    graph_rows = [row for row, _, _ in rows]
    graphs = load_cached_graphs(catalog.get("_catalog_path", ""),
                                graph_rows, graph_field)
    labels = np.asarray([label for _, label, _ in rows])
    test_mask = np.asarray([is_test for _, _, is_test in rows])
    environments = [row.get("environment_group", "unknown") for row, _, _ in rows]
    environment_names = sorted(set(environments))
    environment_ids = np.asarray([environment_names.index(value)
                                  for value in environments], dtype=np.int64)
    return graphs, labels, test_mask, sorted(by_class), environment_ids, environment_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path,
                        default=Path("data/linux_telemetry/reference_candidates_v8/catalog.json"))
    parser.add_argument("--graph-field", choices=("graph_path", "core_graph_path"),
                        default="core_graph_path")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--support-per-class", type=int, default=1)
    parser.add_argument("--similarity-prototype", choices=("support", "all"),
                        default="support",
                        help="support-only or all-training class prototypes")
    parser.add_argument("--metric-weight", type=float, default=0.50)
    parser.add_argument("--margin-weight", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--augmentation", choices=("none",), default="none",
                        help="augmentation backends are not part of this release")
    parser.add_argument("--classifier-C", type=float, default=1.0)
    parser.add_argument("--classifier", choices=("logreg", "svm", "lda"), default="logreg")
    parser.add_argument("--core-pool-weight", type=float, default=0.0,
                        help="evidence-core pooling multiplier; 0 keeps mean pooling")
    parser.add_argument("--pooling", choices=("ordinary", "core", "norm_attention", "logit_attention", "predicted_core", "learned_core"),
                        default="ordinary",
                        help="graph pooling rule; learned_core trains an evidence gate on train masks only")
    parser.add_argument("--core-gate-weight", type=float, default=0.0,
                        help="training-only BCE weight for the learned evidence gate")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--out", type=Path,
                        default=Path("data/experiment_results/coregate_tgn_episode_v8.json"))
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text())
    catalog["_catalog_path"] = args.catalog
    (graphs, labels, test_mask, classes, environment_ids,
     environment_names) = load_rows_with_environment(catalog, args.graph_field)
    class_to_id = {label: index for index, label in enumerate(classes)}
    ids = np.asarray([class_to_id[label] for label in labels])
    train_graphs = [graph for graph, is_test in zip(graphs, test_mask) if not is_test]
    test_graphs = [graph for graph, is_test in zip(graphs, test_mask) if is_test]
    train_labels = labels[~test_mask]
    test_labels = labels[test_mask]
    train_ids = ids[~test_mask].tolist()
    test_ids = ids[test_mask].tolist()
    train_environments = environment_ids[~test_mask].tolist()
    train_data = [base.encoder_input(graph, label)
                  for graph, label in zip(train_graphs, train_ids)]
    test_data = [base.encoder_input(graph, label)
                 for graph, label in zip(test_graphs, test_ids)]

    # Seed before model construction as well as inside the training loop.  The
    # previous protocol seeded only after construction, so two processes with
    # the same reported seed could start from different TGN weights.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = new_model(len(classes))
    last_loss, last_parts, support, query = train_episode_tgn(
        model, train_data, train_ids, args.epochs, args.seed,
        support_per_class=args.support_per_class,
        metric_weight=args.metric_weight, margin_weight=args.margin_weight,
        temperature=args.temperature, margin=args.margin,
        core_pool_weight=args.core_pool_weight,
        pooling=args.pooling,
        core_gate_weight=args.core_gate_weight,
    )
    training_protocol = "episode_tgn"
    train_embeddings = embed_episode_graphs(
        model, train_data, args.core_pool_weight, args.pooling)
    test_embeddings = embed_episode_graphs(
        model, test_data, args.core_pool_weight, args.pooling)
    similarity_support = (all_class_support(train_ids)
                          if args.similarity_prototype == "all" else support)
    train_sim, test_sim = support_similarity(
        train_embeddings, test_embeddings, train_ids, similarity_support)
    train_raw = np.concatenate([train_embeddings.numpy(), train_sim], axis=1)
    test_raw = np.concatenate([test_embeddings.numpy(), test_sim], axis=1)
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    train_features = scaler.fit_transform(train_raw)
    test_features = scaler.transform(test_raw)
    baseline = make_classifier(args.classifier, args.seed, args.classifier_C).fit(
        train_features, train_labels)
    baseline_metrics = metrics(baseline.predict(test_features), test_labels)
    # Only ``--augmentation none`` is supported; the flag and this record are
    # kept because the released commands write them.
    augmentation = {"mode": "none", "synthetic_samples": 0}
    output = {
        "protocol": "coregate_tgn_episode_v1",
        "training_protocol": training_protocol,
        "catalog": str(args.catalog),
        "view": "Core_Graph" if args.graph_field == "core_graph_path" else "Raw_Graph",
        "split": "per_class_last_1_or_2_as_test",
        "seed": args.seed,
        "classes": len(classes),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "embedding_dim": int(train_embeddings.shape[1]),
        "support_per_class": args.support_per_class,
        "similarity_prototype": args.similarity_prototype,
        "episode_query_samples": len(query),
        "epochs": args.epochs,
        "metric_weight": args.metric_weight,
        "margin_weight": args.margin_weight,
        "temperature": args.temperature,
        "margin": args.margin,
        "augmentation_mode": args.augmentation,
        "classifier_C": args.classifier_C,
        "classifier": args.classifier,
        "core_pool_weight": args.core_pool_weight,
        "core_gate_weight": args.core_gate_weight,
        "pooling": args.pooling,
        "environment_groups": environment_names,
        "environment_group_count": len(environment_names),
        "last_loss": last_loss,
        "last_loss_parts": last_parts,
        "tgn_episode_hybrid_logreg": baseline_metrics,
        "augmentation": augmentation,
        "limitations": [
            "the encoder consumes a Linux-compatible 16/8 feature adapter, not raw event streams",
            "support/query episodes are formed from the fixed training split",
            "single deterministic split and seed; diagnostic until multi-seed validation",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in
                      ("protocol", "classes", "train_samples", "test_samples",
                       "episode_query_samples", "tgn_episode_hybrid_logreg")}, indent=2))


if __name__ == "__main__":
    main()
