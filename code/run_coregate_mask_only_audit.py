#!/usr/bin/env python3
"""LOEO audit for simple technique information in weak evidence masks.

The classifier sees only per-graph mask counts, node-type counts, and
operation histograms.  It never sees graph embeddings, environment IDs as
features, or technique labels when constructing the input features.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from coregate.label_blind_lineage import load_mask_records


SEEDS = (20260921, 20260922, 20260923)


def _graph_features(row, operation_vocab, mask_variant="full",
                    lineage_records=None):
    graph = json.loads(Path(row["graph_path"]).read_text())
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    if mask_variant in {"full", "process_only"}:
        core = json.loads(Path(row["core_graph_path"]).read_text())
        core_ids = {node.get("id", index)
                    for index, node in enumerate(core.get("nodes", []))}
    if mask_variant == "full":
        positive_ids = core_ids
        normalize_id = lambda value: value
    elif mask_variant == "process_only":
        positive_ids = {node.get("id") for node in nodes
                        if node.get("id") in core_ids and
                        str(node.get("type", "unknown")) == "process"}
        normalize_id = lambda value: value
    elif mask_variant in {"lineage", "context_union"}:
        if lineage_records is None or row["sample_id"] not in lineage_records:
            raise ValueError(f"missing lineage mask for {row['sample_id']}")
        positive_ids = {
            str(value) for value in
            lineage_records[row["sample_id"]]["positive_node_ids"]
        }
        normalize_id = str
    else:
        raise ValueError(f"unknown mask variant: {mask_variant}")
    positive_nodes = [node for node in nodes
                      if normalize_id(node.get("id")) in positive_ids]
    type_vocab = sorted({str(node.get("type", "unknown")) for node in nodes})
    values = {
        "num_nodes": len(nodes),
        "num_positive_mask_nodes": len(positive_nodes),
        "positive_ratio": len(positive_nodes) / max(1, len(nodes)),
        "num_edges": len(edges),
    }
    for node_type in type_vocab:
        all_count = sum(str(node.get("type", "unknown")) == node_type
                        for node in nodes)
        positive_count = sum(str(node.get("type", "unknown")) == node_type
                             for node in positive_nodes)
        safe = node_type.lower().replace("-", "_").replace(" ", "_")
        values[f"num_{safe}_nodes"] = all_count
        values[f"positive_{safe}_count"] = positive_count
        values[f"positive_{safe}_ratio"] = positive_count / max(1, all_count)
    positive_edges = []
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if (normalize_id(source) in positive_ids and
                normalize_id(target) in positive_ids):
            positive_edges.append(edge)
    values["num_positive_edges"] = len(positive_edges)
    values["positive_edge_ratio"] = len(positive_edges) / max(1, len(edges))
    edge_type_counts = Counter(str(edge.get("operation", "unknown"))
                               for edge in positive_edges)
    for operation in operation_vocab:
        safe = operation.lower().replace("-", "_").replace(" ", "_")
        values[f"positive_edge_operation_{safe}"] = edge_type_counts[operation]
    # Make explicit that no command-line node type is silently omitted.  This
    # catalog encodes commands as process attributes, so this feature is zero;
    # it is retained for a stable audit schema across catalogs.
    values.setdefault("positive_command_count", 0)
    return values


def _metrics(prediction, truth):
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro",
                                    zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path(
        "data/linux_telemetry/reference_candidates_template_v2/catalog.json"))
    parser.add_argument("--out-dir", type=Path, default=Path(
        "data/experiment_results/leave_one_env_v1"))
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--mask-variant", choices=("full", "process_only", "lineage", "context_union"),
                        default="full",
                        help="evidence mask used for the audit features")
    parser.add_argument("--lineage-mask-file", type=Path, default=Path(
                        "data/experiment_results/leave_one_env_v1/label_blind_lineage_masks.json"))
    parser.add_argument("--context-union-mask-file", type=Path, default=Path(
                        "data/experiment_results/leave_one_env_v1/fullrule_context_union_masks.json"))
    parser.add_argument("--output-prefix", default="coregate_mask_only_audit",
                        help="prefix for audit CSV/JSON/Markdown outputs")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    rows = [row for row in catalog["samples"] if row.get("target")]
    environments = sorted({row.get("environment_group") for row in rows})
    operations = sorted({str(edge.get("operation", "unknown"))
                         for row in rows
                         for edge in json.loads(Path(row["graph_path"]).read_text())
                         .get("edges", [])})
    lineage_records = None
    lineage_metadata = None
    if args.mask_variant == "lineage":
        lineage_records, lineage_metadata = load_mask_records(args.lineage_mask_file)
        if not lineage_metadata.get("all_masks_byte_identical_after_label_permutation"):
            raise AssertionError("lineage masks failed the label-invariance audit")
    elif args.mask_variant == "context_union":
        lineage_records, lineage_metadata = load_mask_records(
            args.context_union_mask_file)
        if not lineage_metadata.get("all_full_rule_nodes_retained"):
            raise AssertionError("context union does not retain FullRule")
    features = [_graph_features(row, operations, args.mask_variant,
                               lineage_records) for row in rows]
    feature_names = sorted(features[0])
    matrix = np.asarray([[value[name] for name in feature_names]
                         for value in features], dtype=np.float64)
    labels = np.asarray([row["target"] for row in rows])
    envs = np.asarray([row.get("environment_group") for row in rows])
    results = []
    permutation_results = []
    rng = np.random.default_rng(20260922)
    for seed in SEEDS:
        for heldout in environments:
            train = envs != heldout
            test = ~train
            if set(labels[train]) != set(labels):
                raise AssertionError(f"class coverage failure in {heldout}")
            scaler = StandardScaler()
            train_x = scaler.fit_transform(matrix[train])
            test_x = scaler.transform(matrix[test])
            classifier = LogisticRegression(max_iter=2000,
                                            class_weight="balanced",
                                            multi_class="multinomial",
                                            random_state=seed)
            classifier.fit(train_x, labels[train])
            metric = _metrics(classifier.predict(test_x), labels[test])
            results.append({"seed": seed, "heldout": heldout,
                            "n_train": int(train.sum()),
                            "n_test": int(test.sum()), **metric})
            for permutation in range(args.permutations):
                shuffled = labels[train].copy()
                rng.shuffle(shuffled)
                perm_classifier = LogisticRegression(
                    max_iter=2000, class_weight="balanced",
                    multi_class="multinomial", random_state=seed + permutation)
                perm_classifier.fit(train_x, shuffled)
                perm_metric = _metrics(perm_classifier.predict(test_x), labels[test])
                permutation_results.append({
                    "seed": seed, "heldout": heldout,
                    "permutation": permutation, **perm_metric})

    def summarize(records, key):
        seed_means = [np.mean([r[key] for r in records if r["seed"] == seed])
                      for seed in SEEDS]
        return {"mean": float(np.mean(seed_means)),
                "std_ddof1": float(np.std(seed_means, ddof=1)),
                "seed_means": [float(x) for x in seed_means]}

    summary = {
        "protocol": "coregate_mask_only_label_leakage_audit_v2",
        "catalog": str(args.catalog), "rows": len(rows),
        "mask_variant": args.mask_variant,
        "lineage_mask_file": (str(args.lineage_mask_file)
                              if args.mask_variant == "lineage" else None),
        "context_union_mask_file": (str(args.context_union_mask_file)
                                    if args.mask_variant == "context_union" else None),
        "label_invariance_check": (
            bool(lineage_metadata.get("all_masks_byte_identical_after_label_permutation"))
            if args.mask_variant == "lineage" else None),
        "context_union_retains_full_rule": (
            bool(lineage_metadata.get("all_full_rule_nodes_retained"))
            if args.mask_variant == "context_union" else None),
        "classes": int(len(set(labels.tolist()))), "chance_accuracy": 1 / 45,
        "seeds": list(SEEDS), "folds": environments,
        "feature_names": feature_names, "feature_count": len(feature_names),
        "features_use_embeddings": False, "features_use_environment_id": False,
        "results": results,
        "summary": {"accuracy": summarize(results, "accuracy"),
                    "macro_f1": summarize(results, "macro_f1")},
        "permutation": {
            "n_permutations_per_seed_fold": args.permutations,
            "accuracy": summarize(permutation_results, "accuracy"),
            "macro_f1": summarize(permutation_results, "macro_f1"),
        },
        "limitations": [
            "This audit excludes simple mask-statistic shortcuts; it cannot rule out all label information in richer mask structure.",
            "The catalog has no command-line node type, so positive_command_count is zero; command information remains in process attributes and is not used.",
            "Automatic evidence masks are weak labels pending independent annotation.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.output_prefix}_per_fold.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    perm_path = args.out_dir / f"{args.output_prefix}_permutation.csv"
    with perm_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(permutation_results[0]))
        writer.writeheader(); writer.writerows(permutation_results)
    (args.out_dir / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    lines = [f"# Mask-only label leakage audit ({args.mask_variant})", "",
             f"Rows: {len(rows)}; classes: {summary['classes']}; chance accuracy: 2.22%.",
             "", "| Input | Accuracy | Macro-F1 |", "|---|---:|---:|",
             f"| Mask statistics (real labels) | {summary['summary']['accuracy']['mean']*100:.2f}% ± {summary['summary']['accuracy']['std_ddof1']*100:.2f} | {summary['summary']['macro_f1']['mean']*100:.2f}% ± {summary['summary']['macro_f1']['std_ddof1']*100:.2f} |",
             f"| Mask statistics (100-label permutations) | {summary['permutation']['accuracy']['mean']*100:.2f}% ± {summary['permutation']['accuracy']['std_ddof1']*100:.2f} | {summary['permutation']['macro_f1']['mean']*100:.2f}% ± {summary['permutation']['macro_f1']['std_ddof1']*100:.2f} |",
             "", "The audit uses only mask/node/edge counts and type/operation histograms under the existing four LOEO folds."]
    (args.out_dir / f"{args.output_prefix}.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
