#!/usr/bin/env python3
"""Summarize joint environment/template and edge-rank LOEO controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (20260921, 20260922, 20260923)
METHODS = ("ordinary", "lineage")
MODES = ("ordered", "joint", "shuffled")

_RESULT_DIR = Path("data/experiment_results/leave_one_env_v1")
if not _RESULT_DIR.is_dir():
    # Fresh clone: the shipped artifacts live in results/.
    _RESULT_DIR = Path("results")

STEMS = {
    ("ordered", "ordinary"): "coregate_p1_standard_ordinary",
    ("ordered", "lineage"): "coregate_label_blind_lineage",
    ("joint", "ordinary"): "coregate_joint_ordinary",
    ("joint", "lineage"): "coregate_joint_lineage",
    ("shuffled", "ordinary"): "coregate_order_shuffled_ordinary",
    ("shuffled", "lineage"): "coregate_order_shuffled_lineage",
}


def summarize(rows, key):
    seed_means = [np.mean([row[key] for row in rows if row["seed"] == seed])
                  for seed in SEEDS]
    return {"mean": float(np.mean(seed_means)),
            "std_ddof1": float(np.std(seed_means, ddof=1)),
            "seed_means": [float(value) for value in seed_means]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=_RESULT_DIR)
    parser.add_argument("--output-prefix", default="coregate_joint_order_ablations")
    args = parser.parse_args()

    lookup = {}
    for mode in MODES:
        for method in METHODS:
            for seed in SEEDS:
                path = args.result_dir / f"{STEMS[(mode, method)]}_seed{seed}.json"
                payload = json.loads(path.read_text())
                if len(payload["results"]) != 4:
                    raise AssertionError(f"expected four folds in {path}")
                for fold in payload["results"]:
                    key = (mode, method, seed, fold["holdout_environment"])
                    if key in lookup:
                        raise AssertionError(f"duplicate fold: {key}")
                    if not fold["test_mask_leakage_check"]:
                        raise AssertionError(f"test mask not cleared: {key}")
                    if mode == "joint":
                        if not fold["joint_template_disjoint"] or fold["template_group_overlap_count"]:
                            raise AssertionError(f"joint template leakage: {key}")
                        if fold["support_graph_count"] != 45:
                            raise AssertionError(f"missing joint support class: {key}")
                    elif mode == "shuffled" and fold["edge_time_mode"] != "shuffled":
                        raise AssertionError(f"edge ranks not shuffled: {key}")
                    lookup[key] = fold

    rows = []
    for mode in MODES:
        for seed in SEEDS:
            environments = sorted({key[3] for key in lookup
                                   if key[:3] == (mode, "ordinary", seed)})
            for heldout in environments:
                ordinary = lookup[(mode, "ordinary", seed, heldout)]
                lineage = lookup[(mode, "lineage", seed, heldout)]
                for field in ("train_graph_hash", "test_graph_hash",
                              "support_graph_hash", "train_graph_count",
                              "test_graph_count"):
                    if ordinary[field] != lineage[field]:
                        raise AssertionError(f"method split mismatch: {mode}, {seed}, {heldout}, {field}")
                if mode == "shuffled":
                    for method in METHODS:
                        control = lookup[("ordered", method, seed, heldout)]
                        changed = lookup[(mode, method, seed, heldout)]
                        for field in ("train_graph_hash", "test_graph_hash",
                                      "support_graph_hash"):
                            if control[field] != changed[field]:
                                raise AssertionError(f"edge-order split mismatch: {method}, {seed}, {heldout}")
                if mode == "joint":
                    for method in METHODS:
                        control = lookup[("ordered", method, seed, heldout)]
                        changed = lookup[(mode, method, seed, heldout)]
                        if control["test_graph_hash"] != changed["test_graph_hash"]:
                            raise AssertionError(f"joint test mismatch: {method}, {seed}, {heldout}")
                for method, fold in (("ordinary", ordinary), ("lineage", lineage)):
                    rows.append({
                        "mode": mode, "method": method, "seed": seed,
                        "heldout": heldout, "train_samples": fold["train_samples"],
                        "test_samples": fold["test_samples"],
                        "test_classes": fold["test_class_count"],
                        "train_template_groups": fold.get("train_template_groups"),
                        "test_template_groups": fold.get("test_template_groups"),
                        "template_overlap": fold.get("template_group_overlap_count"),
                        "train_hash": fold["train_graph_hash"],
                        "test_hash": fold["test_graph_hash"],
                        "support_hash": fold["support_graph_hash"],
                        "accuracy": fold["baseline"]["accuracy"],
                        "macro_f1": fold["baseline"]["macro_f1"],
                    })

    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for method in METHODS:
            selected = [row for row in rows if row["mode"] == mode
                        and row["method"] == method]
            summary[mode][method] = {
                metric: summarize(selected, metric)
                for metric in ("accuracy", "macro_f1")
            }
        paired = []
        for row in rows:
            if row["mode"] == mode and row["method"] == "lineage":
                reference = next(item for item in rows
                                 if item["mode"] == mode and
                                 item["method"] == "ordinary" and
                                 item["seed"] == row["seed"] and
                                 item["heldout"] == row["heldout"])
                paired.append({"seed": row["seed"],
                               "accuracy": row["accuracy"] - reference["accuracy"],
                               "macro_f1": row["macro_f1"] - reference["macro_f1"]})
        summary[mode]["lineage_minus_ordinary"] = {
            metric: {**summarize(paired, metric),
                     "positive_pairs": sum(item[metric] > 0 for item in paired),
                     "total_pairs": len(paired)}
            for metric in ("accuracy", "macro_f1")
        }
        if mode != "ordered":
            summary[mode]["minus_ordered"] = {}
            for method in METHODS:
                changes = []
                for seed in SEEDS:
                    for heldout in sorted({key[3] for key in lookup
                                           if key[:3] == (mode, method, seed)}):
                        current = lookup[(mode, method, seed, heldout)]
                        control = lookup[("ordered", method, seed, heldout)]
                        changes.append({
                            "seed": seed,
                            **{metric: current["baseline"][metric] -
                               control["baseline"][metric]
                               for metric in ("accuracy", "macro_f1")},
                        })
                summary[mode]["minus_ordered"][method] = {
                    metric: {**summarize(changes, metric),
                             "positive_pairs": sum(item[metric] > 0 for item in changes),
                             "negative_pairs": sum(item[metric] < 0 for item in changes)}
                    for metric in ("accuracy", "macro_f1")
                }

    output = {
        "protocol": "coregate_joint_environment_template_and_edge_rank_ablation_v1",
        "seeds": list(SEEDS), "fold_count": 4,
        "methods": list(METHODS), "modes": list(MODES),
        "summary": summary, "fold_rows": rows,
        "interpretation_boundary": [
            "Joint holdout removes all training graphs sharing a template group with the held-out environment; test graphs remain unchanged.",
            "Joint folds have sharply different training sizes, so their means are a pressure test, not an isolated template-effect estimate.",
            "The Linux adapter uses normalized edge-list rank as edge_time, not original wall-clock timestamps.",
            "Shuffling edge_time values per graph preserves topology, operation attributes, and the edge-time multiset while breaking their alignment.",
            "All model comparisons clear held-out core masks before inference; lineage masks supervise only training graphs.",
        ],
    }
    prefix = args.result_dir / args.output_prefix
    prefix.with_name(prefix.name + "_summary.json").write_text(
        json.dumps(output, indent=2) + "\n")
    with prefix.with_name(prefix.name + "_per_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Joint environment-template and edge-rank ablations", "",
             "Four LOEO folds, three seeds, 15 epochs, StandardScaler, matched ordinary TGN and label-blind CoreGate. Held-out masks are cleared.",
             "", "| Protocol | Ordinary accuracy / F1 | Lineage accuracy / F1 | Paired gain accuracy / F1 | Positive pairs |",
             "|---|---:|---:|---:|---:|"]
    for mode in MODES:
        ordinary = summary[mode]["ordinary"]
        lineage = summary[mode]["lineage"]
        gain = summary[mode]["lineage_minus_ordinary"]
        lines.append(
            f"| {mode} | {100*ordinary['accuracy']['mean']:.2f}% / {100*ordinary['macro_f1']['mean']:.2f}% | "
            f"{100*lineage['accuracy']['mean']:.2f}% / {100*lineage['macro_f1']['mean']:.2f}% | "
            f"{100*gain['accuracy']['mean']:+.2f} / {100*gain['macro_f1']['mean']:+.2f} pp | "
            f"{gain['accuracy']['positive_pairs']}/12 accuracy, {gain['macro_f1']['positive_pairs']}/12 F1 |")
    lines.extend(["", "## Difference from ordered LOEO", ""])
    for mode in ("joint", "shuffled"):
        for method in METHODS:
            change = summary[mode]["minus_ordered"][method]
            lines.append(
                f"- {mode} {method}: {100*change['accuracy']['mean']:+.2f} pp accuracy, "
                f"{100*change['macro_f1']['mean']:+.2f} pp macro-F1; "
                f"negative in {change['accuracy']['negative_pairs']}/12 and "
                f"{change['macro_f1']['negative_pairs']}/12 paired folds.")
    lines.extend(["", "Joint holdout removes training graphs that share any template group with the held-out environment. Its training sets are substantially smaller and vary by fold. The shuffled condition keeps topology, edge operations, and each graph's normalized edge-rank values, but permutes their edge assignment; these are rank positions rather than raw timestamps.",
                  "", "All joint method pairs share train/test/support hashes; shuffled and ordered pairs also share those hashes. All held-out masks are cleared. The per-fold CSV and JSON contain exact scores, counts, and hashes."])
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
