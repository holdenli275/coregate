#!/usr/bin/env python3
"""Summarize label-blind positive-dropout sensitivity experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from coregate.label_blind_lineage import load_mask_records


BASE = Path("data/experiment_results/leave_one_env_v1")
if not BASE.is_dir():
        # Fresh clone: the shipped artifacts live in results/.
    BASE = Path("results")
SEED = 20260921
SEEDS = (20260921, 20260922, 20260923)
METHODS = {
    "ordinary": "coregate_p1_standard_ordinary_seed20260921.json",
    "lineage_clean": "coregate_label_blind_lineage_seed20260921.json",
    "lineage_drop10": "coregate_label_blind_noise_drop10_seed20260921.json",
    "lineage_drop30": "coregate_label_blind_noise_drop30_seed20260921.json",
    "lineage_drop50": "coregate_label_blind_noise_drop50_seed20260921.json",
}
NOISE_FILES = {
    "lineage_drop10": "label_blind_lineage_drop10.json",
    "lineage_drop30": "label_blind_lineage_drop30.json",
    "lineage_drop50": "label_blind_lineage_drop50.json",
}
DROP50_MASK_FILES = {
    20260921: "label_blind_lineage_drop50.json",
    20260922: "label_blind_lineage_drop50_seed20260922.json",
    20260923: "label_blind_lineage_drop50_seed20260923.json",
}


def _load(path):
    return json.loads((BASE / path).read_text())


def _paired(rows, left, right, metric):
    reference = {row["heldout"]: row for row in rows
                 if row["method"] == right}
    values = [row[metric] - reference[row["heldout"]][metric]
              for row in rows if row["method"] == left]
    return {
        "mean": float(np.mean(values)), "min": float(np.min(values)),
        "max": float(np.max(values)),
        "positive_count": int(sum(value > 0 for value in values)),
        "nonnegative_count": int(sum(value >= 0 for value in values)),
        "n": len(values), "values": values,
    }


def _paired_multi(rows, left, right, metric):
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
    loaded = {method: _load(path)["results"]
              for method, path in METHODS.items()}
    reference = loaded["ordinary"]
    reference_keys = [(row["holdout_environment"], row["train_graph_hash"],
                       row["test_graph_hash"], row["support_graph_hash"])
                      for row in reference]
    rows = []
    gated_parameters = set()
    for method, results in loaded.items():
        keys = [(row["holdout_environment"], row["train_graph_hash"],
                 row["test_graph_hash"], row["support_graph_hash"])
                for row in results]
        if keys != reference_keys:
            raise AssertionError(f"split/support mismatch for {method}")
        for result in results:
            if result["classifier_scaler"] != "standard":
                raise AssertionError("classifier scaler mismatch")
            if not result["test_mask_leakage_check"]:
                raise AssertionError("test-time mask leakage")
            if method != "ordinary":
                gated_parameters.add(result["trainable_parameters"])
                if result["train_core_mask"] != "lineage":
                    raise AssertionError("unexpected gate supervision source")
                if not result["lineage_label_invariance_check"]:
                    raise AssertionError("label-invariance check failed")
                if result["train_lineage_positive_count_max_abs_delta"] != 0:
                    raise AssertionError("serialized mask count mismatch")
            rows.append({
                "method": method,
                "seed": SEED,
                "heldout": result["holdout_environment"],
                "n_test": result["test_samples"],
                "accuracy": result["baseline"]["accuracy"],
                "macro_f1": result["baseline"]["macro_f1"],
                "train_graph_hash": result["train_graph_hash"],
                "test_graph_hash": result["test_graph_hash"],
                "support_graph_hash": result["support_graph_hash"],
                "test_mask_leakage_check": result["test_mask_leakage_check"],
            })
    if len(gated_parameters) != 1:
        raise AssertionError("gated parameter counts differ")

    noise = {method: _load(path) for method, path in NOISE_FILES.items()}
    clean, _ = load_mask_records(BASE / "label_blind_lineage_masks.json")
    previous = {sample_id: set(record["positive_node_ids"])
                for sample_id, record in clean.items()}
    for method in ("lineage_drop10", "lineage_drop30", "lineage_drop50"):
        current = {sample_id: set(record["positive_node_ids"])
                   for sample_id, record in noise[method]["masks"].items()}
        if any(not current[sample_id].issubset(previous[sample_id])
               for sample_id in current):
            raise AssertionError("dropout masks are not nested")
        previous = current

    summary = {
        method: {
            metric: float(np.mean([row[metric] for row in rows
                                   if row["method"] == method]))
            for metric in ("accuracy", "macro_f1")
        } for method in METHODS
    }
    comparisons = {}
    for method in METHODS:
        if method != "ordinary":
            comparisons[f"{method}_minus_ordinary"] = {
                metric: _paired(rows, method, "ordinary", metric)
                for metric in ("accuracy", "macro_f1")
            }
        if method.startswith("lineage_drop"):
            comparisons[f"{method}_minus_lineage_clean"] = {
                metric: _paired(rows, method, "lineage_clean", metric)
                for metric in ("accuracy", "macro_f1")
            }

    multi_rows = []
    multi_parameters = set()
    multi_noise_audit = {}
    for seed in SEEDS:
        multi_loaded = {
            "ordinary": _load(
                f"coregate_p1_standard_ordinary_seed{seed}.json")["results"],
            "lineage_clean": _load(
                f"coregate_label_blind_lineage_seed{seed}.json")["results"],
            "lineage_drop50": _load(
                f"coregate_label_blind_noise_drop50_seed{seed}.json")["results"],
        }
        ref_keys = [
            (row["holdout_environment"], row["train_graph_hash"],
             row["test_graph_hash"], row["support_graph_hash"])
            for row in multi_loaded["ordinary"]
        ]
        for method, results in multi_loaded.items():
            keys = [(row["holdout_environment"], row["train_graph_hash"],
                     row["test_graph_hash"], row["support_graph_hash"])
                    for row in results]
            if keys != ref_keys:
                raise AssertionError(
                    f"three-seed split/support mismatch for {method}/{seed}")
            for result in results:
                if result["classifier_scaler"] != "standard":
                    raise AssertionError("three-seed scaler mismatch")
                if not result["test_mask_leakage_check"]:
                    raise AssertionError("three-seed test-time mask leakage")
                if method != "ordinary":
                    multi_parameters.add(result["trainable_parameters"])
                    if not result["lineage_label_invariance_check"]:
                        raise AssertionError("three-seed label invariance failure")
                    if result["train_lineage_positive_count_max_abs_delta"] != 0:
                        raise AssertionError("three-seed mask count mismatch")
                multi_rows.append({
                    "method": method, "seed": seed,
                    "heldout": result["holdout_environment"],
                    "n_test": result["test_samples"],
                    "accuracy": result["baseline"]["accuracy"],
                    "macro_f1": result["baseline"]["macro_f1"],
                    "train_graph_hash": result["train_graph_hash"],
                    "test_graph_hash": result["test_graph_hash"],
                    "support_graph_hash": result["support_graph_hash"],
                    "test_mask_leakage_check": result["test_mask_leakage_check"],
                })
        mask_metadata = _load(DROP50_MASK_FILES[seed])
        if mask_metadata["noise_seed"] != seed:
            raise AssertionError("corruption seed mismatch")
        current = {sample_id: set(record["positive_node_ids"])
                   for sample_id, record in mask_metadata["masks"].items()}
        if any(not current[sample_id].issubset(
                set(clean[sample_id]["positive_node_ids"]))
               for sample_id in current):
            raise AssertionError("50% mask is not a clean-mask subset")
        multi_noise_audit[str(seed)] = {
            key: value for key, value in mask_metadata.items()
            if key != "masks"
        }
    if len(multi_parameters) != 1:
        raise AssertionError("three-seed gated parameter counts differ")

    def _three_seed_metric(method, metric):
        seed_values = [float(np.mean([
            row[metric] for row in multi_rows
            if row["method"] == method and row["seed"] == seed
        ])) for seed in SEEDS]
        return {
            "mean": float(np.mean(seed_values)),
            "std_ddof1": float(np.std(seed_values, ddof=1)),
            "seed_values": seed_values,
        }

    multi_summary = {
        method: {metric: _three_seed_metric(method, metric)
                 for metric in ("accuracy", "macro_f1")}
        for method in ("ordinary", "lineage_clean", "lineage_drop50")
    }
    multi_comparisons = {
        "lineage_drop50_minus_ordinary": {
            metric: _paired_multi(
                multi_rows, "lineage_drop50", "ordinary", metric)
            for metric in ("accuracy", "macro_f1")
        },
        "lineage_drop50_minus_lineage_clean": {
            metric: _paired_multi(
                multi_rows, "lineage_drop50", "lineage_clean", metric)
            for metric in ("accuracy", "macro_f1")
        },
    }

    output = {
        "protocol": "label_blind_lineage_positive_dropout_sensitivity_v1",
        "scope": "single model seed and single deterministic corruption seed",
        "seed": SEED, "folds": 4, "epochs": 15,
        "summary": summary, "paired_comparisons": comparisons,
        "noise_audit": {
            method: {key: value for key, value in metadata.items()
                     if key != "masks"}
            for method, metadata in noise.items()
        },
        "sanity_checks": {
            "same_graph_and_support_hashes": True,
            "all_test_masks_cleared": True,
            "all_dropout_masks_nested": True,
            "all_dropout_masks_nonempty": True,
            "gated_trainable_parameters": next(iter(gated_parameters)),
        },
        "interpretation_limit": (
            "This is a one-seed sensitivity analysis, not a new primary "
            "multi-seed estimate."),
        "drop50_three_seed": {
            "scope": "three model seeds with matched deterministic corruption seeds",
            "seeds": list(SEEDS), "folds": 4, "epochs": 15,
            "summary": multi_summary,
            "paired_comparisons": multi_comparisons,
            "noise_audit": multi_noise_audit,
            "sanity_checks": {
                "paired_rows": len(multi_rows),
                "same_graph_and_support_hashes": True,
                "all_test_masks_cleared": True,
                "all_dropout_masks_are_clean_mask_subsets": True,
                "all_dropout_masks_nonempty": True,
                "gated_trainable_parameters": next(iter(multi_parameters)),
            },
        },
    }
    with (BASE / "coregate_label_blind_noise_sensitivity_per_fold.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (BASE / "coregate_label_blind_noise_drop50_three_seed_per_fold.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(multi_rows[0]))
        writer.writeheader()
        writer.writerows(multi_rows)
    (BASE / "coregate_label_blind_noise_sensitivity_summary.json").write_text(
        json.dumps(output, indent=2) + "\n")

    labels = {
        "ordinary": "Ordinary", "lineage_clean": "Lineage clean",
        "lineage_drop10": "Lineage drop 10%",
        "lineage_drop30": "Lineage drop 30%",
        "lineage_drop50": "Lineage drop 50%",
    }
    def multi_cell(method, metric):
        item = multi_summary[method][metric]
        return f"{item['mean']*100:.2f}% +/- {item['std_ddof1']*100:.2f}"

    multi_gain = multi_comparisons["lineage_drop50_minus_ordinary"]
    lines = [
        "# Label-blind mask-noise sensitivity", "",
        "## Three-seed 50% positive-dropout result", "",
        "Three matched model seeds and deterministic corruption seeds; four LOEO folds, 15 epochs.", "",
        "| Condition | Accuracy | Macro-F1 |", "|---|---:|---:|",
        f"| Ordinary | {multi_cell('ordinary', 'accuracy')} | {multi_cell('ordinary', 'macro_f1')} |",
        f"| Lineage clean | {multi_cell('lineage_clean', 'accuracy')} | {multi_cell('lineage_clean', 'macro_f1')} |",
        f"| Lineage drop 50% | {multi_cell('lineage_drop50', 'accuracy')} | {multi_cell('lineage_drop50', 'macro_f1')} |",
        "",
        f"Drop 50% minus Ordinary: {multi_gain['accuracy']['mean']*100:+.2f} Accuracy / {multi_gain['macro_f1']['mean']*100:+.2f} Macro-F1; positive in {multi_gain['accuracy']['positive_count']}/12 and {multi_gain['macro_f1']['positive_count']}/12 paired runs.",
        "", "## Single-seed noise curve", "",
        "Single model seed and deterministic corruption seed; four LOEO folds, 15 epochs.", "",
        "| Condition | Accuracy | Macro-F1 |", "|---|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            f"| {labels[method]} | {summary[method]['accuracy']*100:.2f}% | "
            f"{summary[method]['macro_f1']*100:.2f}% |")
    lines.extend(["", "## Paired gain over Ordinary", ""])
    for method in METHODS:
        if method == "ordinary":
            continue
        comparison = comparisons[f"{method}_minus_ordinary"]
        lines.append(
            f"- {labels[method]}: {comparison['accuracy']['mean']*100:+.2f} "
            f"Accuracy / {comparison['macro_f1']['mean']*100:+.2f} Macro-F1; "
            f"positive folds {comparison['accuracy']['positive_count']}/4 and "
            f"{comparison['macro_f1']['positive_count']}/4.")
    lines.extend([
        "", "## Interpretation", "",
        "Deleting approximately 10%, 30%, or 50% of positive training-mask nodes does not remove the gain over Ordinary in the fixed-seed curve. The strict 50% condition is additionally confirmed over three matched model and corruption seeds. This supports tolerance to false-negative foreground annotations; it remains a supplementary robustness result rather than independent-host validation.",
    ])
    (BASE / "coregate_label_blind_noise_sensitivity.md").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
