#!/usr/bin/env python3
"""Summarize the conservative process-only CoreGate sensitivity run."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (20260921, 20260922, 20260923)
BASE = Path("data/experiment_results/leave_one_env_v1")
if not BASE.is_dir():
        # Fresh clone: the shipped artifacts live in results/.
    BASE = Path("results")


def main() -> None:
    rows = []
    seed_summaries = []
    for outer_seed in SEEDS:
        process = json.loads(
            (BASE / f"coregate_process_only_seed{outer_seed}.json").read_text()
        )["results"]
        ordinary = json.loads(
            (BASE / f"coregate_p1_standard_ordinary_seed{outer_seed}.json").read_text()
        )["results"]
        if [x["holdout_environment"] for x in process] != [
            x["holdout_environment"] for x in ordinary
        ]:
            raise AssertionError("process-only and ordinary folds do not match")
        for p, o in zip(process, ordinary):
            rows.append({
                "outer_seed": outer_seed,
                "fold_seed": p["seed"],
                "heldout": p["holdout_environment"],
                "n_test": p["test_samples"],
                "ordinary_accuracy": o["baseline"]["accuracy"],
                "ordinary_macro_f1": o["baseline"]["macro_f1"],
                "process_only_accuracy": p["baseline"]["accuracy"],
                "process_only_macro_f1": p["baseline"]["macro_f1"],
                "accuracy_gain": p["baseline"]["accuracy"] - o["baseline"]["accuracy"],
                "macro_f1_gain": p["baseline"]["macro_f1"] - o["baseline"]["macro_f1"],
                "train_core_mask": p["train_core_mask"],
                "train_positive_count_max_abs_delta": p["train_core_positive_count_max_abs_delta"],
                "test_mask_leakage_check": p["test_mask_leakage_check"],
            })
        seed_summaries.append({
            "outer_seed": outer_seed,
            "ordinary_accuracy": float(np.mean([x["ordinary_accuracy"] for x in rows if x["outer_seed"] == outer_seed])),
            "ordinary_macro_f1": float(np.mean([x["ordinary_macro_f1"] for x in rows if x["outer_seed"] == outer_seed])),
            "process_only_accuracy": float(np.mean([x["process_only_accuracy"] for x in rows if x["outer_seed"] == outer_seed])),
            "process_only_macro_f1": float(np.mean([x["process_only_macro_f1"] for x in rows if x["outer_seed"] == outer_seed])),
        })

    def summary(key):
        values = np.asarray([x[key] for x in seed_summaries], dtype=float)
        return {"mean": float(values.mean()), "std_ddof1": float(values.std(ddof=1)),
                "seed_values": values.tolist()}

    acc_gains = np.asarray([x["accuracy_gain"] for x in rows])
    f1_gains = np.asarray([x["macro_f1_gain"] for x in rows])
    output = {
        "protocol": "coregate_process_only_conservative_mask_v1",
        "mask_definition": "serialized evidence core intersected with process nodes only",
        "catalog": "data/linux_telemetry/reference_candidates_template_v2/catalog.json",
        "folds": 4, "seeds": list(SEEDS), "epochs": 15,
        "classifier_scaler": "standard", "test_core_mask": "none",
        "shared_gate_forward": True,
        "seed_summaries": seed_summaries,
        "summary": {
            "ordinary_accuracy": summary("ordinary_accuracy"),
            "ordinary_macro_f1": summary("ordinary_macro_f1"),
            "process_only_accuracy": summary("process_only_accuracy"),
            "process_only_macro_f1": summary("process_only_macro_f1"),
        },
        "paired_gain": {
            "accuracy": {"mean": float(acc_gains.mean()), "min": float(acc_gains.min()),
                          "max": float(acc_gains.max()), "positive_count": int((acc_gains > 0).sum()),
                          "n": int(len(acc_gains))},
            "macro_f1": {"mean": float(f1_gains.mean()), "min": float(f1_gains.min()),
                         "max": float(f1_gains.max()), "positive_count": int((f1_gains > 0).sum()),
                         "n": int(len(f1_gains))},
        },
        "sanity_checks": {
            "all_process_only_sources": all(x["train_core_mask"] == "process_only" for x in rows),
            "max_train_positive_count_delta": max(x["train_positive_count_max_abs_delta"] for x in rows),
            "all_test_masks_cleared": all(x["test_mask_leakage_check"] for x in rows),
        },
        "p0_process_only_mask_audit": json.loads(
            (BASE / "coregate_process_only_mask_audit_summary.json").read_text()
        ),
        "caveat": "This supplementary run uses the shared-forward path used by P1/P2; dropout call order differs from the locked main run.",
    }
    csv_path = BASE / "coregate_process_only_per_fold.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (BASE / "coregate_process_only_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    md = [
        "# Conservative process-only mask sensitivity",
        "",
        "The training evidence target is the serialized evidence core intersected with process nodes only. File, socket, and pipe evidence nodes are excluded; held-out masks are cleared at inference.",
        "",
        "| Method | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
        f"| Ordinary | {output['summary']['ordinary_accuracy']['mean']*100:.2f}% +/- {output['summary']['ordinary_accuracy']['std_ddof1']*100:.2f} | {output['summary']['ordinary_macro_f1']['mean']*100:.2f}% +/- {output['summary']['ordinary_macro_f1']['std_ddof1']*100:.2f} |",
        f"| Process-only CoreGate | {output['summary']['process_only_accuracy']['mean']*100:.2f}% +/- {output['summary']['process_only_accuracy']['std_ddof1']*100:.2f} | {output['summary']['process_only_macro_f1']['mean']*100:.2f}% +/- {output['summary']['process_only_macro_f1']['std_ddof1']*100:.2f} |",
        "",
        f"Paired gain over Ordinary: +{acc_gains.mean()*100:.2f} pp Accuracy (min {acc_gains.min()*100:.2f}, max {acc_gains.max()*100:.2f}; {int((acc_gains > 0).sum())}/12 positive) and +{f1_gains.mean()*100:.2f} pp Macro-F1 (min {f1_gains.min()*100:.2f}, max {f1_gains.max()*100:.2f}; {int((f1_gains > 0).sum())}/12 positive).",
        "",
        f"The process-only mask statistics alone obtain {output['p0_process_only_mask_audit']['summary']['accuracy']['mean']*100:.2f}% Accuracy and {output['p0_process_only_mask_audit']['summary']['macro_f1']['mean']*100:.2f}% Macro-F1, versus 91.41% / 91.36% for the full mask audit.",
        "",
        "This is a sensitivity analysis, not a replacement for the locked full-evidence result. It supports using process-only evidence as a more conservative configuration while retaining a technique-specific weak-prior caveat.",
    ]
    (BASE / "coregate_process_only.md").write_text("\n".join(md) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
