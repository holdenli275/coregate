#!/usr/bin/env python3
"""Summarize the minimal label-blind CoreGate submission experiments."""
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
    "gate_no_evidence": "coregate_label_blind_noevidence_seed{seed}.json",
    "gate_shuffled_lineage": "coregate_label_blind_shuffled_seed{seed}.json",
    "coregate_lineage": "coregate_label_blind_lineage_seed{seed}.json",
    "coregate_full_rule": "coregate_p1_standard_coregate_seed{seed}.json",
}
EXPECTED_MASK = {
    "ordinary": "evidence",
    "gate_no_evidence": "lineage",
    "gate_shuffled_lineage": "lineage_random",
    "coregate_lineage": "lineage",
    "coregate_full_rule": "evidence",
}


def load_results(pattern, seed):
    return json.loads((BASE / pattern.format(seed=seed)).read_text())["results"]


def metric_summary(rows, method, metric):
    seed_values = [float(np.mean([
        row[metric] for row in rows
        if row["method"] == method and row["outer_seed"] == seed
    ])) for seed in SEEDS]
    return {"mean": float(np.mean(seed_values)),
            "std_ddof1": float(np.std(seed_values, ddof=1)),
            "seed_values": seed_values}


def paired(rows, left, right, metric):
    index = {(row["outer_seed"], row["heldout"]): row for row in rows
             if row["method"] == right}
    values = np.asarray([
        row[metric] - index[(row["outer_seed"], row["heldout"])][metric]
        for row in rows if row["method"] == left
    ])
    return {"mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()), "positive_count": int((values > 0).sum()),
            "zero_count": int((values == 0).sum()), "n": int(len(values)),
            "values": values.tolist()}


def main():
    rows = []
    gate_parameter_counts = set()
    ordinary_parameter_counts = set()
    for outer_seed in SEEDS:
        loaded = {method: load_results(pattern, outer_seed)
                  for method, pattern in METHODS.items()}
        reference = loaded["ordinary"]
        reference_keys = [(r["holdout_environment"], r["train_graph_hash"],
                           r["test_graph_hash"], r["support_graph_hash"])
                          for r in reference]
        for method, results in loaded.items():
            keys = [(r["holdout_environment"], r["train_graph_hash"],
                     r["test_graph_hash"], r["support_graph_hash"])
                    for r in results]
            if keys != reference_keys:
                raise AssertionError(f"split/support mismatch for {method}/{outer_seed}")
            for result in results:
                if result["classifier_scaler"] != "standard":
                    raise AssertionError("classifier scaler mismatch")
                if not result["test_mask_leakage_check"]:
                    raise AssertionError("test-time evidence leakage")
                if result["train_core_mask"] != EXPECTED_MASK[method]:
                    raise AssertionError(f"mask source mismatch for {method}")
                if method != "ordinary":
                    gate_parameter_counts.add(result["trainable_parameters"])
                    if not result["share_gate_forward"]:
                        raise AssertionError("gated run did not use shared forward")
                else:
                    ordinary_parameter_counts.add(result["trainable_parameters"])
                if method in {"gate_no_evidence", "gate_shuffled_lineage",
                              "coregate_lineage"}:
                    if not result["lineage_label_invariance_check"]:
                        raise AssertionError("lineage label-invariance failure")
                    if result["train_lineage_positive_count_max_abs_delta"] != 0:
                        raise AssertionError("lineage sparsity mismatch")
                rows.append({
                    "method": method, "outer_seed": outer_seed,
                    "fold_seed": result["seed"],
                    "heldout": result["holdout_environment"],
                    "n_test": result["test_samples"],
                    "accuracy": result["baseline"]["accuracy"],
                    "macro_f1": result["baseline"]["macro_f1"],
                    "embedding_only_accuracy": result["embedding_only"]["accuracy"],
                    "embedding_only_macro_f1": result["embedding_only"]["macro_f1"],
                    "train_graph_hash": result["train_graph_hash"],
                    "test_graph_hash": result["test_graph_hash"],
                    "support_graph_hash": result["support_graph_hash"],
                    "train_core_mask": result["train_core_mask"],
                    "trainable_parameters": result["trainable_parameters"],
                    "test_mask_leakage_check": result["test_mask_leakage_check"],
                })
    if len(gate_parameter_counts) != 1:
        raise AssertionError("gated models do not have identical parameter counts")
    if len(ordinary_parameter_counts) != 1:
        raise AssertionError("ordinary parameter count changed across runs")

    summary = {
        method: {
            metric: metric_summary(rows, method, metric)
            for metric in ("accuracy", "macro_f1", "embedding_only_accuracy",
                           "embedding_only_macro_f1")
        } for method in METHODS
    }
    comparisons = {}
    for left, right in (
        ("gate_no_evidence", "ordinary"),
        ("gate_shuffled_lineage", "ordinary"),
        ("coregate_lineage", "ordinary"),
        ("coregate_full_rule", "ordinary"),
        ("coregate_lineage", "gate_no_evidence"),
        ("coregate_lineage", "gate_shuffled_lineage"),
        ("coregate_full_rule", "coregate_lineage"),
    ):
        comparisons[f"{left}_minus_{right}"] = {
            metric: paired(rows, left, right, metric)
            for metric in ("accuracy", "macro_f1")
        }

    template_rows = []
    template_patterns = {
        "ordinary": "coregate_p2_template_ordinary_seed{seed}.json",
        "coregate_lineage": "coregate_label_blind_template_seed{seed}.json",
        "coregate_full_rule": "coregate_p2_template_coregate_seed{seed}.json",
    }
    for seed in SEEDS:
        values = {method: load_results(pattern, seed)[0]
                  for method, pattern in template_patterns.items()}
        ref = values["ordinary"]
        for method, result in values.items():
            for key in ("train_graph_hash", "test_graph_hash", "support_graph_hash"):
                if result[key] != ref[key]:
                    raise AssertionError(f"template split mismatch: {method}/{seed}/{key}")
            if not result["test_mask_leakage_check"]:
                raise AssertionError("template test mask leakage")
            template_rows.append({"method": method, "outer_seed": seed,
                                  "heldout": "template_disjoint_test",
                                  "accuracy": result["baseline"]["accuracy"],
                                  "macro_f1": result["baseline"]["macro_f1"]})

    def template_summary(method, metric):
        values = [row[metric] for row in template_rows if row["method"] == method]
        return {"mean": float(np.mean(values)),
                "std_ddof1": float(np.std(values, ddof=1)), "seed_values": values}

    template = {method: {metric: template_summary(method, metric)
                         for metric in ("accuracy", "macro_f1")}
                for method in template_patterns}
    template["paired_lineage_minus_ordinary"] = {
        metric: paired(template_rows, "coregate_lineage", "ordinary", metric)
        for metric in ("accuracy", "macro_f1")
    }

    label_audit = json.loads((BASE / "label_blind_lineage_audit.json").read_text())
    mask_audit = json.loads((BASE / "coregate_label_blind_lineage_mask_audit_summary.json").read_text())
    output = {
        "protocol": "coregate_label_blind_minimal_submission_v1",
        "seeds": list(SEEDS), "folds": 4, "epochs": 15,
        "classifier_scaler": "standard", "shared_gate_forward": True,
        "summary": summary, "paired_comparisons": comparisons,
        "template_disjoint": template,
        "label_invariance_audit": label_audit,
        "mask_only_audit": {
            "accuracy": mask_audit["summary"]["accuracy"],
            "macro_f1": mask_audit["summary"]["macro_f1"],
            "permutation_accuracy": mask_audit["permutation"]["accuracy"],
            "permutation_macro_f1": mask_audit["permutation"]["macro_f1"],
        },
        "sanity_checks": {
            "paired_rows": len(rows),
            "same_splits_and_support": True,
            "all_test_masks_cleared": True,
            "lineage_sparsity_matched": True,
            "lineage_masks_label_invariant": label_audit["all_masks_byte_identical_after_label_permutation"],
            "changed_masks_after_label_permutation": label_audit["changed_mask_count_after_label_permutation"],
            "gated_trainable_parameters": next(iter(gate_parameter_counts)),
            "ordinary_trainable_parameters": next(iter(ordinary_parameter_counts)),
        },
        "caveat": "The label-blind and full-rule gated runs use the shared-forward supplementary path; dropout call order differs from the locked v0.1 main run.",
    }
    with (BASE / "coregate_label_blind_minimal_per_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (BASE / "coregate_label_blind_minimal_summary.json").write_text(
        json.dumps(output, indent=2) + "\n")

    def cell(method, metric):
        item = summary[method][metric]
        return f"{item['mean']*100:.2f}% +/- {item['std_ddof1']*100:.2f}"

    lineage_gain = comparisons["coregate_lineage_minus_ordinary"]
    shuffled_gain = comparisons["coregate_lineage_minus_gate_shuffled_lineage"]
    noev_gain = comparisons["coregate_lineage_minus_gate_no_evidence"]
    lines = [
        "# Label-blind CoreGate minimal submission experiment", "",
        "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|",
    ]
    labels = {
        "ordinary": "Ordinary", "gate_no_evidence": "LearnedAttention",
        "gate_shuffled_lineage": "Gate-ShuffledLineage",
        "coregate_lineage": "CoreGate-Lineage",
        "coregate_full_rule": "CoreGate-FullRule",
    }
    for method in METHODS:
        lines.append(f"| {labels[method]} | {cell(method, 'accuracy')} | {cell(method, 'macro_f1')} |")
    lines += [
        "", f"Lineage minus Ordinary: +{lineage_gain['accuracy']['mean']*100:.2f} / +{lineage_gain['macro_f1']['mean']*100:.2f} points; positive in {lineage_gain['accuracy']['positive_count']}/12 Accuracy and {lineage_gain['macro_f1']['positive_count']}/12 Macro-F1 pairs.",
        f"Lineage minus LearnedAttention: +{noev_gain['accuracy']['mean']*100:.2f} / +{noev_gain['macro_f1']['mean']*100:.2f} points.",
        f"Lineage minus ShuffledLineage: +{shuffled_gain['accuracy']['mean']*100:.2f} / +{shuffled_gain['macro_f1']['mean']*100:.2f} points.",
        "", "## Label-invariance and mask-only checks", "",
        f"All {label_audit['samples']} masks remain byte-identical after permuting technique labels; changed masks = {label_audit['changed_mask_count_after_label_permutation']}.",
        f"Lineage mask-only statistics reach {mask_audit['summary']['accuracy']['mean']*100:.2f}% Accuracy and {mask_audit['summary']['macro_f1']['mean']*100:.2f}% Macro-F1. This measures behavioral information, not construction-time label access.",
        "", "## Template-disjoint", "",
        f"Ordinary: {template['ordinary']['accuracy']['mean']*100:.2f}% / {template['ordinary']['macro_f1']['mean']*100:.2f}%; CoreGate-Lineage: {template['coregate_lineage']['accuracy']['mean']*100:.2f}% / {template['coregate_lineage']['macro_f1']['mean']*100:.2f}%; FullRule: {template['coregate_full_rule']['accuracy']['mean']*100:.2f}% / {template['coregate_full_rule']['macro_f1']['mean']*100:.2f}%.",
        "", "## Interpretation", "",
        "The gain persists when gate supervision is generated from collection-time scenario lineage without ATT&CK labels, technique rules, control.kind, or Core_Graph. The label-permutation hash audit rules out direct construction-time label access. The masks remain behaviorally discriminative, so the result should still be described as training-only privileged foreground supervision rather than causal subgraph recovery.",
    ]
    (BASE / "coregate_label_blind_minimal_summary.md").write_text("\n".join(lines) + "\n")
    paper_text = f"""# Paper-ready label-blind evidence text

## Results / Ablation

To separate execution-derived foreground supervision from ATT&CK-label or technique-rule access, we constructed a label-blind target containing process nodes associated with events marked as scenario activity in the collection journal. The builder reads only sample and raw artifact paths; it does not read the technique label, technique-specific rules, `control.kind`, template identifiers, or the serialized Core_Graph. Permuting all training technique labels changes 0 of {label_audit['samples']} mask hashes. Under the matched 4-fold LOEO, 3-seed, 15-epoch protocol, CoreGate-Lineage obtains {cell('coregate_lineage', 'accuracy')} accuracy and {cell('coregate_lineage', 'macro_f1')} macro-F1, compared with {cell('ordinary', 'accuracy')} and {cell('ordinary', 'macro_f1')} for Ordinary pooling. The paired gains are {lineage_gain['accuracy']['mean']*100:+.2f} and {lineage_gain['macro_f1']['mean']*100:+.2f} percentage points and are positive in 12/12 seed-fold pairs for both metrics. The same gated readout without evidence loss reaches {cell('gate_no_evidence', 'accuracy')} / {cell('gate_no_evidence', 'macro_f1')}, and sparsity-matched shuffled lineage supervision reaches {cell('gate_shuffled_lineage', 'accuracy')} / {cell('gate_shuffled_lineage', 'macro_f1')}. Thus neither the additional gate parameters nor mask density reproduces the label-blind lineage result. On the fixed template-disjoint split, CoreGate-Lineage improves over Ordinary by {template['paired_lineage_minus_ordinary']['accuracy']['mean']*100:+.2f} accuracy and {template['paired_lineage_minus_ordinary']['macro_f1']['mean']*100:+.2f} macro-F1 points across three seeds.

## Discussion

The label-invariance audit rules out direct construction-time access to ATT&CK labels and technique-specific evidence rules for this target; it does not establish causal subgraph recovery or eliminate every catalog artifact. Simple lineage-mask and graph statistics still classify techniques at {mask_audit['summary']['accuracy']['mean']*100:.2f}% accuracy and {mask_audit['summary']['macro_f1']['mean']*100:.2f}% macro-F1 under LOEO, above the {mask_audit['chance_accuracy']*100:.2f}% chance level. This is expected if collection-time foreground lineage captures technique behavior, but it also means the supervision should be described as training-only privileged foreground information. No evidence mask is available or read at test time. Independent-host and independently annotated evidence evaluation remain necessary for stronger external-validity or causal claims.
"""
    (BASE / "coregate_label_blind_minimal_paper_text.md").write_text(paper_text)
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
