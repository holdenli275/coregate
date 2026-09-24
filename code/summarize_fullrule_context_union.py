#!/usr/bin/env python3
"""Summarize the FullRule union label-blind context sensitivity."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


BASE = Path("data/experiment_results/leave_one_env_v1")
if not BASE.is_dir():
        # Fresh clone: the shipped artifacts live in results/.
    BASE = Path("results")
SEEDS = (20260921, 20260922, 20260923)
METHODS = {
    "ordinary": "coregate_p1_standard_ordinary_seed{seed}.json",
    "coregate_full_rule": "coregate_p1_standard_coregate_seed{seed}.json",
    "coregate_lineage": "coregate_label_blind_lineage_seed{seed}.json",
    "fullrule_context_union": "coregate_fullrule_context_union_seed{seed}.json",
}


def _load(path):
    return json.loads((BASE / path).read_text())


def _metric_summary(rows, method, metric):
    seed_values = [float(np.mean([
        row[metric] for row in rows
        if row["method"] == method and row["seed"] == seed
    ])) for seed in SEEDS]
    return {
        "mean": float(np.mean(seed_values)),
        "std_ddof1": float(np.std(seed_values, ddof=1)),
        "seed_values": seed_values,
    }


def _paired(rows, left, right, metric):
    reference = {(row["seed"], row["heldout"]): row for row in rows
                 if row["method"] == right}
    values = [row[metric] - reference[(row["seed"], row["heldout"])][metric]
              for row in rows if row["method"] == left]
    return {
        "mean": float(np.mean(values)), "min": float(np.min(values)),
        "max": float(np.max(values)),
        "positive_count": int(sum(value > 0 for value in values)),
        "nonnegative_count": int(sum(value >= 0 for value in values)),
        "n": len(values), "values": values,
    }


def main():
    rows = []
    parameter_counts = {method: set() for method in METHODS}
    for seed in SEEDS:
        loaded = {
            method: _load(pattern.format(seed=seed))["results"]
            for method, pattern in METHODS.items()
        }
        reference_keys = [
            (row["holdout_environment"], row["train_graph_hash"],
             row["test_graph_hash"], row["support_graph_hash"])
            for row in loaded["ordinary"]
        ]
        for method, results in loaded.items():
            keys = [(row["holdout_environment"], row["train_graph_hash"],
                     row["test_graph_hash"], row["support_graph_hash"])
                    for row in results]
            if keys != reference_keys:
                raise AssertionError(f"split/support mismatch: {method}/{seed}")
            for result in results:
                if result["classifier_scaler"] != "standard":
                    raise AssertionError("classifier scaler mismatch")
                if not result["test_mask_leakage_check"]:
                    raise AssertionError("test-time evidence leakage")
                if method == "fullrule_context_union":
                    if result["train_core_mask"] != "context_union":
                        raise AssertionError("context-union mask source mismatch")
                    if not result["context_union_retains_full_rule"]:
                        raise AssertionError("context union lost FullRule nodes")
                    if result["train_context_union_positive_count_max_abs_delta"] != 0:
                        raise AssertionError("context-union positive count mismatch")
                parameter_counts[method].add(result["trainable_parameters"])
                rows.append({
                    "method": method, "seed": seed,
                    "heldout": result["holdout_environment"],
                    "n_test": result["test_samples"],
                    "accuracy": result["baseline"]["accuracy"],
                    "macro_f1": result["baseline"]["macro_f1"],
                    "embedding_only_accuracy": result["embedding_only"]["accuracy"],
                    "embedding_only_macro_f1": result["embedding_only"]["macro_f1"],
                    "train_graph_hash": result["train_graph_hash"],
                    "test_graph_hash": result["test_graph_hash"],
                    "support_graph_hash": result["support_graph_hash"],
                    "trainable_parameters": result["trainable_parameters"],
                    "test_mask_leakage_check": result["test_mask_leakage_check"],
                })
    if any(len(values) != 1 for values in parameter_counts.values()):
        raise AssertionError("parameter count changed within a method")

    summary = {
        method: {metric: _metric_summary(rows, method, metric)
                 for metric in ("accuracy", "macro_f1",
                                "embedding_only_accuracy",
                                "embedding_only_macro_f1")}
        for method in METHODS
    }
    comparisons = {}
    for left, right in (
        ("fullrule_context_union", "ordinary"),
        ("fullrule_context_union", "coregate_full_rule"),
        ("fullrule_context_union", "coregate_lineage"),
    ):
        comparisons[f"{left}_minus_{right}"] = {
            metric: _paired(rows, left, right, metric)
            for metric in ("accuracy", "macro_f1")
        }

    construction = _load("fullrule_context_union_masks.json")
    full_mask_audit = _load("coregate_mask_only_audit_summary.json")
    union_mask_audit = _load(
        "coregate_fullrule_context_union_mask_audit_summary.json")
    mask_only = {
        "full_rule": {
            metric: full_mask_audit["summary"][metric]
            for metric in ("accuracy", "macro_f1")
        },
        "context_union": {
            metric: union_mask_audit["summary"][metric]
            for metric in ("accuracy", "macro_f1")
        },
    }
    mask_only["context_union_minus_full_rule"] = {
        metric: (mask_only["context_union"][metric]["mean"] -
                 mask_only["full_rule"][metric]["mean"])
        for metric in ("accuracy", "macro_f1")
    }

    output = {
        "protocol": "fullrule_context_union_sensitivity_v1",
        "seeds": list(SEEDS), "folds": 4, "epochs": 15,
        "summary": summary, "paired_comparisons": comparisons,
        "mask_only_audit": mask_only,
        "construction_audit": {
            key: value for key, value in construction.items()
            if key != "masks"
        },
        "sanity_checks": {
            "rows": len(rows), "same_graph_and_support_hashes": True,
            "all_test_masks_cleared": True,
            "all_full_rule_nodes_retained": True,
            "context_union_positive_count_delta": 0,
            "parameter_counts": {
                method: next(iter(values))
                for method, values in parameter_counts.items()
            },
        },
        "decision": "reject_context_union_as_shortcut_mitigation",
        "interpretation": (
            "ContextUnion lowers mean CoreGate classification relative to "
            "FullRule but increases mask-only predictiveness. The lower model "
            "score therefore cannot be interpreted as reduced rule shortcut."),
    }
    with (BASE / "coregate_fullrule_context_union_per_fold.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (BASE / "coregate_fullrule_context_union_summary.json").write_text(
        json.dumps(output, indent=2) + "\n")

    def cell(method, metric):
        item = summary[method][metric]
        return f"{item['mean']*100:.2f}% +/- {item['std_ddof1']*100:.2f}"

    context_vs_ordinary = comparisons[
        "fullrule_context_union_minus_ordinary"]
    context_vs_full = comparisons[
        "fullrule_context_union_minus_coregate_full_rule"]
    lines = [
        "# FullRule-ContextUnion sensitivity", "",
        "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|",
        f"| Ordinary | {cell('ordinary', 'accuracy')} | {cell('ordinary', 'macro_f1')} |",
        f"| CoreGate-FullRule | {cell('coregate_full_rule', 'accuracy')} | {cell('coregate_full_rule', 'macro_f1')} |",
        f"| CoreGate-Lineage | {cell('coregate_lineage', 'accuracy')} | {cell('coregate_lineage', 'macro_f1')} |",
        f"| FullRule-ContextUnion | {cell('fullrule_context_union', 'accuracy')} | {cell('fullrule_context_union', 'macro_f1')} |",
        "", "## Mask-only audit", "",
        f"FullRule: {mask_only['full_rule']['accuracy']['mean']*100:.2f}% Accuracy / {mask_only['full_rule']['macro_f1']['mean']*100:.2f}% Macro-F1.",
        f"ContextUnion: {mask_only['context_union']['accuracy']['mean']*100:.2f}% Accuracy / {mask_only['context_union']['macro_f1']['mean']*100:.2f}% Macro-F1.",
        f"Change: {mask_only['context_union_minus_full_rule']['accuracy']*100:+.2f} / {mask_only['context_union_minus_full_rule']['macro_f1']*100:+.2f} points.",
        "", "## Paired model results", "",
        f"ContextUnion minus Ordinary: {context_vs_ordinary['accuracy']['mean']*100:+.2f} Accuracy / {context_vs_ordinary['macro_f1']['mean']*100:+.2f} Macro-F1; positive in {context_vs_ordinary['accuracy']['positive_count']}/12 and {context_vs_ordinary['macro_f1']['positive_count']}/12 pairs.",
        f"ContextUnion minus FullRule: {context_vs_full['accuracy']['mean']*100:+.2f} Accuracy / {context_vs_full['macro_f1']['mean']*100:+.2f} Macro-F1.",
        "", "## Decision", "",
        "Reject ContextUnion as shortcut mitigation. It retains all FullRule nodes and adds label-blind process context, but the mask-only fingerprint becomes stronger rather than weaker. Its lower mean CoreGate result is therefore consistent with noisier gate supervision, not evidence that the rule shortcut was reduced.",
    ]
    (BASE / "coregate_fullrule_context_union.md").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
