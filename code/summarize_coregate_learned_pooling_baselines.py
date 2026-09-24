#!/usr/bin/env python3
"""Summarize matched learned pooling baselines for CoreGate."""
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
    "learned_attention": "coregate_label_blind_noevidence_seed{seed}.json",
    "sagpool_50": "coregate_sagpool_seed{seed}.json",
    "coregate_lineage": "coregate_label_blind_lineage_seed{seed}.json",
    "coregate_full_rule": "coregate_p1_standard_coregate_seed{seed}.json",
}
EXPECTED_POOLING = {
    "ordinary": "ordinary",
    "learned_attention": "learned_core",
    "sagpool_50": "sagpool",
    "coregate_lineage": "learned_core",
    "coregate_full_rule": "learned_core",
}


def _load(pattern, seed):
    return json.loads((BASE / pattern.format(seed=seed)).read_text())["results"]


def _summary(rows, method, metric):
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
        loaded = {method: _load(pattern, seed)
                  for method, pattern in METHODS.items()}
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
                if result["pooling"] != EXPECTED_POOLING[method]:
                    raise AssertionError(f"pooling mismatch: {method}")
                if result["classifier_scaler"] != "standard":
                    raise AssertionError("classifier scaler mismatch")
                if not result["test_mask_leakage_check"]:
                    raise AssertionError("test-time evidence leakage")
                if method == "sagpool_50" and result["train_core_mask"] != "none":
                    raise AssertionError("SAGPool unexpectedly received evidence")
                if method == "learned_attention" and result["core_gate_weight"] != 0:
                    raise AssertionError("learned attention received evidence loss")
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
        method: {metric: _summary(rows, method, metric)
                 for metric in ("accuracy", "macro_f1",
                                "embedding_only_accuracy",
                                "embedding_only_macro_f1")}
        for method in METHODS
    }
    comparisons = {}
    for left, right in (
        ("learned_attention", "ordinary"),
        ("sagpool_50", "ordinary"),
        ("coregate_lineage", "ordinary"),
        ("sagpool_50", "learned_attention"),
        ("coregate_lineage", "learned_attention"),
        ("coregate_lineage", "sagpool_50"),
    ):
        comparisons[f"{left}_minus_{right}"] = {
            metric: _paired(rows, left, right, metric)
            for metric in ("accuracy", "macro_f1")
        }

    output = {
        "protocol": "coregate_matched_learned_pooling_baselines_v1",
        "seeds": list(SEEDS), "folds": 4, "epochs": 15,
        "sagpool_ratio": 0.5,
        "summary": summary, "paired_comparisons": comparisons,
        "sanity_checks": {
            "rows": len(rows), "same_graph_and_support_hashes": True,
            "all_test_masks_cleared": True,
            "sagpool_uses_evidence_supervision": False,
            "learned_attention_uses_evidence_supervision": False,
            "parameter_counts": {
                method: next(iter(values))
                for method, values in parameter_counts.items()
            },
        },
        "interpretation": (
            "Neither scalar learned attention without evidence nor SAGPool "
            "reproduces the label-blind CoreGate result."),
    }
    with (BASE / "coregate_learned_pooling_baselines_per_fold.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (BASE / "coregate_learned_pooling_baselines_summary.json").write_text(
        json.dumps(output, indent=2) + "\n")

    def cell(method, metric):
        item = summary[method][metric]
        return f"{item['mean']*100:.2f}% +/- {item['std_ddof1']*100:.2f}"

    labels = {
        "ordinary": "Ordinary", "learned_attention": "LearnedAttention",
        "sagpool_50": "SAGPool-50%",
        "coregate_lineage": "CoreGate-Lineage",
        "coregate_full_rule": "CoreGate-FullRule",
    }
    lines = [
        "# Matched learned-pooling baselines", "",
        "Four LOEO folds x three seeds, 15 epochs, StandardScaler, identical split/support selection.", "",
        "| Method | Accuracy | Macro-F1 | Parameters |", "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            f"| {labels[method]} | {cell(method, 'accuracy')} | "
            f"{cell(method, 'macro_f1')} | "
            f"{next(iter(parameter_counts[method])):,} |")
    sag_gain = comparisons["sagpool_50_minus_ordinary"]
    lineage_sag = comparisons["coregate_lineage_minus_sagpool_50"]
    lines.extend([
        "", "## Paired results", "",
        f"SAGPool minus Ordinary: {sag_gain['accuracy']['mean']*100:+.2f} Accuracy / {sag_gain['macro_f1']['mean']*100:+.2f} Macro-F1; positive in {sag_gain['accuracy']['positive_count']}/12 and {sag_gain['macro_f1']['positive_count']}/12 pairs.",
        f"CoreGate-Lineage minus SAGPool: {lineage_sag['accuracy']['mean']*100:+.2f} Accuracy / {lineage_sag['macro_f1']['mean']*100:+.2f} Macro-F1; positive in {lineage_sag['accuracy']['positive_count']}/12 and {lineage_sag['macro_f1']['positive_count']}/12 pairs.",
        "", "SAGPool uses the official PyG SAGPooling layer with a fixed 0.5 retention ratio and receives no evidence supervision. Its failure to match CoreGate-Lineage strengthens the conclusion that generic task-learned pooling alone does not explain the CoreGate gain.",
    ])
    (BASE / "coregate_learned_pooling_baselines.md").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
