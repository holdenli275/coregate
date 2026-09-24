#!/usr/bin/env python3
"""Summarize the StandardScaler P1 and fixed template-disjoint P2 runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (20260921, 20260922, 20260923)
METHODS = ("ordinary", "coregate")

_RESULT_DIR = Path("data/experiment_results/leave_one_env_v1")
if not _RESULT_DIR.is_dir():
    # Fresh clone: the shipped artifacts live in results/.
    _RESULT_DIR = Path("results")



def _load(directory, prefix, method, seed):
    path = directory / f"{prefix}_{method}_seed{seed}.json"
    obj = json.loads(path.read_text())
    return path, obj


def _summarize(rows, method, value):
    seed_means = []
    for seed in SEEDS:
        subset = [r[value] for r in rows if r["method"] == method and r["seed"] == seed]
        seed_means.append(float(np.mean(subset)))
    return {"mean": float(np.mean(seed_means)),
            "std_ddof1": float(np.std(seed_means, ddof=1)),
            "seed_means": seed_means}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=_RESULT_DIR)
    parser.add_argument("--p2-audit", type=Path, default=
        _RESULT_DIR / "template_disjoint_minimal_split.json")
    args = parser.parse_args()

    p1 = []
    p2 = []
    for method in METHODS:
        for seed in SEEDS:
            path, obj = _load(args.results_dir, "coregate_p1_standard", method, seed)
            for result in obj["results"]:
                p1.append({"method": method, "seed": seed,
                           "heldout": result["holdout_environment"],
                           "n_test": result["test_samples"],
                           "accuracy_embedding_only": result["embedding_only"]["accuracy"],
                           "macro_f1_embedding_only": result["embedding_only"]["macro_f1"],
                           "accuracy_embedding_plus_proto": result["baseline"]["accuracy"],
                           "macro_f1_embedding_plus_proto": result["baseline"]["macro_f1"],
                           "classifier_scaler": result.get("classifier_scaler"),
                           "share_gate_forward": result.get("share_gate_forward", False),
                           "train_graph_hash": result["train_graph_hash"],
                           "test_graph_hash": result["test_graph_hash"],
                           "support_graph_hash": result["support_graph_hash"],
                           "test_mask_leakage_check": result["test_mask_leakage_check"],
                           "source_file": path.name})
            path, obj = _load(args.results_dir, "coregate_p2_template", method, seed)
            for result in obj["results"]:
                p2.append({"method": method, "seed": seed,
                           "heldout": result["holdout_environment"],
                           "n_train": result["train_samples"],
                           "n_test": result["test_samples"],
                           "test_class_count": result["test_class_count"],
                           "accuracy": result["baseline"]["accuracy"],
                           "macro_f1": result["baseline"]["macro_f1"],
                           "accuracy_embedding_only": result["embedding_only"]["accuracy"],
                           "macro_f1_embedding_only": result["embedding_only"]["macro_f1"],
                           "classifier_scaler": result.get("classifier_scaler"),
                           "share_gate_forward": result.get("share_gate_forward", False),
                           "train_graph_hash": result["train_graph_hash"],
                           "test_graph_hash": result["test_graph_hash"],
                           "support_graph_hash": result["support_graph_hash"],
                           "test_mask_leakage_check": result["test_mask_leakage_check"],
                           "source_file": path.name})
    if len(p1) != 24 or len(p2) != 6:
        raise AssertionError(f"expected 24 P1 rows and 6 P2 rows, got {len(p1)} and {len(p2)}")
    if {r["classifier_scaler"] for r in p1 + p2} != {"standard"}:
        raise AssertionError("P1/P2 must use StandardScaler")
    for rows, label in ((p1, "P1"), (p2, "P2")):
        for r in rows:
            expected = r["method"] == "coregate"
            if r["share_gate_forward"] != expected:
                raise AssertionError(f"{label} gate-forward protocol mismatch")
    for seed in SEEDS:
        for heldout in sorted({r["heldout"] for r in p1 if r["seed"] == seed}):
            hashes = {(r["train_graph_hash"], r["test_graph_hash"], r["support_graph_hash"])
                      for r in p1 if r["seed"] == seed and r["heldout"] == heldout}
            if len(hashes) != 1:
                raise AssertionError(f"P1 split mismatch for seed {seed}, {heldout}")
    if any(not r["test_mask_leakage_check"] for r in p1 + p2):
        raise AssertionError("test mask leakage check failed")
    audit = json.loads(args.p2_audit.read_text())
    if audit["template_group_overlap_count"] != 0:
        raise AssertionError("P2 template overlap is nonzero")
    p2_hashes = {(r["train_graph_hash"], r["test_graph_hash"], r["support_graph_hash"])
                 for r in p2}
    if len(p2_hashes) != 1:
        raise AssertionError("P2 split/support hashes differ")

    summary = {"protocol": "coregate_p1_p2_minimal_v1",
               "p1": {"rows": len(p1), "ordinary": {}, "coregate": {},
                      "paired_coregate_minus_ordinary": {}},
               "p2": {"rows": len(p2), "ordinary": {}, "coregate": {},
                      "template_split": audit,
                      "paired_coregate_minus_ordinary": {}}}
    for method in METHODS:
        for value in ("accuracy_embedding_only", "macro_f1_embedding_only",
                      "accuracy_embedding_plus_proto", "macro_f1_embedding_plus_proto"):
            summary["p1"][method][value] = _summarize(p1, method, value)
        for value in ("accuracy", "macro_f1", "accuracy_embedding_only",
                      "macro_f1_embedding_only"):
            summary["p2"][method][value] = _summarize(p2, method, value)
    for value in ("accuracy_embedding_only", "macro_f1_embedding_only",
                  "accuracy_embedding_plus_proto", "macro_f1_embedding_plus_proto"):
        gains = [r[value] - next(x[value] for x in p1
                                 if x["method"] == "ordinary" and x["seed"] == r["seed"] and x["heldout"] == r["heldout"])
                 for r in p1 if r["method"] == "coregate"]
        summary["p1"]["paired_coregate_minus_ordinary"][value] = {
            "mean": float(np.mean(gains)), "min": float(np.min(gains)),
            "max": float(np.max(gains)), "positive_count": int(sum(x > 0 for x in gains)),
            "n": len(gains)}
    for value in ("accuracy", "macro_f1", "accuracy_embedding_only", "macro_f1_embedding_only"):
        gains = [r[value] - next(x[value] for x in p2
                                 if x["method"] == "ordinary" and x["seed"] == r["seed"])
                 for r in p2 if r["method"] == "coregate"]
        summary["p2"]["paired_coregate_minus_ordinary"][value] = {
            "mean": float(np.mean(gains)), "min": float(np.min(gains)),
            "max": float(np.max(gains)), "positive_count": int(sum(x > 0 for x in gains)),
            "n": len(gains)}

    output = args.results_dir / "coregate_p1_p2_minimal_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    with (args.results_dir / "coregate_p1_per_fold.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(p1[0])); writer.writeheader(); writer.writerows(p1)
    with (args.results_dir / "coregate_p2_per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(p2[0])); writer.writeheader(); writer.writerows(p2)
    lines = ["# CoreGate P1/P2 minimal supplementary experiments", "",
             "P1 uses StandardScaler and the same encoder checkpoint for embedding-only and embedding-plus-prototype inputs. CoreGate uses the supplementary shared node-representation forward path to avoid repeating the gate BCE encoder call; this path preserves the objective but changes dropout call order relative to the locked main run.", "",
             "| Method | P1 embedding-only accuracy | P1 embedding+prototype accuracy | P1 embedding-only Macro-F1 |", "|---|---:|---:|---:|"]
    for method in METHODS:
        s = summary["p1"][method]
        lines.append(f"| {method} | {s['accuracy_embedding_only']['mean']*100:.2f}% ± {s['accuracy_embedding_only']['std_ddof1']*100:.2f} | {s['accuracy_embedding_plus_proto']['mean']*100:.2f}% ± {s['accuracy_embedding_plus_proto']['std_ddof1']*100:.2f} | {s['macro_f1_embedding_only']['mean']*100:.2f}% ± {s['macro_f1_embedding_only']['std_ddof1']*100:.2f} |")
    p1g = summary["p1"]["paired_coregate_minus_ordinary"]
    lines.append(f"\nP1 CoreGate − Ordinary: embedding-only accuracy {p1g['accuracy_embedding_only']['mean']*100:.2f} pp, Macro-F1 {p1g['macro_f1_embedding_only']['mean']*100:.2f} pp, both positive in {p1g['accuracy_embedding_only']['positive_count']}/{p1g['accuracy_embedding_only']['n']} paired runs; embedding+prototype accuracy {p1g['accuracy_embedding_plus_proto']['mean']*100:.2f} pp and Macro-F1 {p1g['macro_f1_embedding_plus_proto']['mean']*100:.2f} pp.")
    lines += ["", f"P2 uses one fixed held-out template group per class; train/test template overlap is {audit['template_group_overlap_count']}.", "", "| Method | P2 accuracy | P2 Macro-F1 |", "|---|---:|---:|"]
    for method in METHODS:
        s = summary["p2"][method]
        lines.append(f"| {method} | {s['accuracy']['mean']*100:.2f}% ± {s['accuracy']['std_ddof1']*100:.2f} | {s['macro_f1']['mean']*100:.2f}% ± {s['macro_f1']['std_ddof1']*100:.2f} |")
    p2g = summary["p2"]["paired_coregate_minus_ordinary"]
    lines.append(f"\nP2 CoreGate − Ordinary: accuracy {p2g['accuracy']['mean']*100:.2f} pp, Macro-F1 {p2g['macro_f1']['mean']*100:.2f} pp; positive in {p2g['accuracy']['positive_count']}/{p2g['accuracy']['n']} seeds for both metrics.")
    (args.results_dir / "coregate_p1_p2_minimal_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
