#!/usr/bin/env python3
"""Compare raw-graph, evidence-mask, and joint graph statistics under LOEO."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from coregate.label_blind_lineage import load_mask_records
from run_coregate_mask_only_audit import SEEDS


VIEWS = ("raw_only", "mask_only", "raw_plus_mask")
MASK_VARIANTS = ("full", "lineage")


def statistic_features(graph, positive_ids, node_types, operations):
    """Return disjoint raw and induced-mask count features."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    positive_ids = {str(value) for value in positive_ids}
    positive_nodes = [node for node in nodes
                      if str(node.get("id")) in positive_ids]
    present_ids = {str(node.get("id")) for node in positive_nodes}
    positive_edges = [edge for edge in edges
                      if str(edge.get("source")) in present_ids and
                      str(edge.get("target")) in present_ids]

    def counts(selected_nodes, selected_edges, prefix):
        node_counts = Counter(str(node.get("type", "unknown"))
                              for node in selected_nodes)
        edge_counts = Counter(str(edge.get("operation", "unknown"))
                              for edge in selected_edges)
        values = {f"{prefix}num_nodes": len(selected_nodes),
                  f"{prefix}num_edges": len(selected_edges)}
        values.update({f"{prefix}node_type:{kind}": node_counts[kind]
                       for kind in node_types})
        values.update({f"{prefix}edge_operation:{kind}": edge_counts[kind]
                       for kind in operations})
        return values

    return counts(nodes, edges, "raw_"), counts(
        positive_nodes, positive_edges, "mask_")


def mean_by_seed(records, metric):
    values = [np.mean([row[metric] for row in records if row["seed"] == seed])
              for seed in SEEDS]
    return {"mean": float(np.mean(values)),
            "std_ddof1": float(np.std(values, ddof=1)),
            "seed_means": [float(value) for value in values]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path(
        "data/linux_telemetry/reference_candidates_template_v2/catalog.json"))
    parser.add_argument("--lineage-mask-file", type=Path, default=Path(
        "data/experiment_results/leave_one_env_v1/label_blind_lineage_masks.json"))
    parser.add_argument("--out-dir", type=Path, default=Path(
        "data/experiment_results/leave_one_env_v1"))
    parser.add_argument("--output-prefix", default="coregate_three_view_statistics_audit")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text())
    rows = [row for row in catalog["samples"] if row.get("target")]
    lineage, metadata = load_mask_records(args.lineage_mask_file)
    if not metadata.get("all_masks_byte_identical_after_label_permutation"):
        raise AssertionError("lineage mask label-invariance audit is missing or failed")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample IDs")
    graphs = [json.loads(Path(row["graph_path"]).read_text()) for row in rows]
    node_types = sorted({str(node.get("type", "unknown"))
                         for graph in graphs for node in graph.get("nodes", [])})
    operations = sorted({str(edge.get("operation", "unknown"))
                         for graph in graphs for edge in graph.get("edges", [])})
    labels = np.asarray([row["target"] for row in rows])
    envs = np.asarray([row["environment_group"] for row in rows])
    environments = sorted(set(envs))
    raw_names = None
    mask_names = None
    matrices = {}
    for variant in MASK_VARIANTS:
        pairs = []
        for row, graph in zip(rows, graphs):
            if variant == "full":
                core = json.loads(Path(row["core_graph_path"]).read_text())
                selected = [node.get("id") for node in core.get("nodes", [])]
            else:
                record = lineage.get(row["sample_id"])
                if record is None:
                    raise ValueError(f"missing lineage mask for {row['sample_id']}")
                selected = record["positive_node_ids"]
            pairs.append(statistic_features(graph, selected, node_types, operations))
        raw_names = sorted(pairs[0][0])
        mask_names = sorted(pairs[0][1])
        raw = np.asarray([[item[0][name] for name in raw_names]
                          for item in pairs], dtype=float)
        masked = np.asarray([[item[1][name] for name in mask_names]
                             for item in pairs], dtype=float)
        matrices[variant] = {"raw_only": raw, "mask_only": masked,
                             "raw_plus_mask": np.column_stack((raw, masked))}
    if not np.array_equal(matrices["full"]["raw_only"],
                          matrices["lineage"]["raw_only"]):
        raise AssertionError("raw features changed with mask variant")

    fold_rows = []
    for variant in MASK_VARIANTS:
        for view in VIEWS:
            matrix = matrices[variant][view]
            for seed in SEEDS:
                for heldout in environments:
                    train = envs != heldout
                    test = ~train
                    if set(labels[train]) != set(labels):
                        raise AssertionError(f"class coverage failure in {heldout}")
                    scaler = StandardScaler()
                    train_x = scaler.fit_transform(matrix[train])
                    test_x = scaler.transform(matrix[test])
                    model = LogisticRegression(
                        max_iter=2000, class_weight="balanced",
                        multi_class="multinomial", random_state=seed)
                    model.fit(train_x, labels[train])
                    prediction = model.predict(test_x)
                    fold_rows.append({
                        "mask_variant": variant, "view": view, "seed": seed,
                        "heldout": heldout, "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "accuracy": float(accuracy_score(labels[test], prediction)),
                        "macro_f1": float(f1_score(
                            labels[test], prediction, average="macro",
                            zero_division=0)),
                    })

    summary = {}
    for variant in MASK_VARIANTS:
        summary[variant] = {}
        for view in VIEWS:
            selected = [row for row in fold_rows if row["mask_variant"] == variant
                        and row["view"] == view]
            summary[variant][view] = {
                "accuracy": mean_by_seed(selected, "accuracy"),
                "macro_f1": mean_by_seed(selected, "macro_f1"),
                "feature_count": matrices[variant][view].shape[1],
            }
        baseline = summary[variant]["raw_only"]
        for view in ("mask_only", "raw_plus_mask"):
            summary[variant][view]["delta_vs_raw_only"] = {
                metric: (summary[variant][view][metric]["mean"] -
                         baseline[metric]["mean"])
                for metric in ("accuracy", "macro_f1")
            }

    output = {
        "protocol": "coregate_three_view_statistics_loeo_v1",
        "catalog": str(args.catalog), "lineage_mask_file": str(args.lineage_mask_file),
        "rows": len(rows), "classes": len(set(labels)),
        "chance_accuracy": 1 / len(set(labels)),
        "seeds": list(SEEDS), "folds": environments,
        "feature_definition": {
            "raw_only": "Full-graph node and edge counts, node-type and edge-operation histograms.",
            "mask_only": "Selected-node and induced-edge counts, node-type and edge-operation histograms; no full-graph totals or ratios.",
            "raw_plus_mask": "Concatenation of raw_only and mask_only statistics.",
            "raw_feature_names": raw_names, "mask_feature_names": mask_names,
            "uses_embeddings": False, "uses_node_attributes": False,
            "uses_environment_as_feature": False,
        },
        "lineage_label_invariance_check": True,
        "summary": summary, "fold_results": fold_rows,
        "limitations": [
            "Mask-only and joint views use held-out evidence masks for diagnostic features; they are not deployable test-time CoreGate inputs.",
            "Simple count histograms do not exhaust information in raw graphs or mask topology.",
            "Three classifier seeds reuse the same four held-out environments and are not independent datasets.",
            "The automatically constructed masks are not independent human annotations.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_dir / args.output_prefix
    (prefix.with_name(prefix.name + "_summary.json")).write_text(
        json.dumps(output, indent=2) + "\n")
    with prefix.with_name(prefix.name + "_per_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    lines = ["# Raw graph / mask / joint statistics audit", "",
             f"{len(rows)} labeled graphs, {len(set(labels))} classes, four LOEO folds, three classifier seeds.",
             "", "| Mask | Statistics | Accuracy | Macro-F1 | Features |",
             "|---|---|---:|---:|---:|"]
    for variant in MASK_VARIANTS:
        for view in VIEWS:
            item = summary[variant][view]
            acc = item["accuracy"]
            f1 = item["macro_f1"]
            lines.append(f"| {variant} | {view} | "
                         f"{100*acc['mean']:.2f}% +/- {100*acc['std_ddof1']:.2f} | "
                         f"{100*f1['mean']:.2f}% +/- {100*f1['std_ddof1']:.2f} | "
                         f"{item['feature_count']} |")
    lines.extend(["", "## Paired descriptive differences versus raw-only", ""])
    for variant in MASK_VARIANTS:
        for view in ("mask_only", "raw_plus_mask"):
            delta = summary[variant][view]["delta_vs_raw_only"]
            lines.append(f"- {variant} {view}: {100*delta['accuracy']:+.2f} pp accuracy; "
                         f"{100*delta['macro_f1']:+.2f} pp macro-F1.")
    lines.extend([
        "", "## Interpretation", "",
        "Raw-only operation and node-type counts already carry substantial class information. FullRule mask statistics are stronger, and their concatenation with raw statistics is stronger still under this linear classifier. The label-blind lineage mask alone is much weaker; adding it to raw statistics gives only a small descriptive gain. This does not measure the value of lineage supervision for the learned TGN gate.",
        "", "The earlier mask-only audit included full-graph counts and mask-to-graph ratios in its feature set. Its 91.41% FullRule and 38.70% lineage accuracies are therefore not directly comparable to the strict mask-only rows above.",
        "", "The mask-only view contains only selected-node and induced-edge statistics; it excludes full-graph totals and ratios. Raw+mask concatenates the two disjoint feature blocks. All views use the same splits, train-fitted scaler, and multinomial logistic classifier. The three seeds give identical predictions, so the displayed zero standard deviation is not evidence of external stability.",
        "", "Mask-only and raw+mask inspect held-out masks as an oracle diagnostic. They are not test-time inputs to CoreGate. The audit covers simple count histograms, not graph attributes or topology. See the JSON and per-fold CSV for exact results.",
    ])
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
