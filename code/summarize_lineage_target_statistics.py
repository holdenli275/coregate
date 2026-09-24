#!/usr/bin/env python3
"""Descriptive statistics of the execution-lineage targets quoted in the paper.

Reads the shipped lineage masks and the catalog, and recomputes the target-size
statistics reported in the paper: mean/median/quartiles and range of positive
nodes per graph, whether the largest targets belong to one technique, and the
average fraction of process nodes selected per environment fold.

Needs neither the dataset archive nor a GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RESULT_DIR = Path("data/experiment_results/leave_one_env_v1")
if not RESULT_DIR.is_dir():
    # Fresh clone: the shipped artifacts live in results/.
    RESULT_DIR = Path("results")
MASK_FILE = RESULT_DIR / "label_blind_lineage_masks.json"
CATALOG_FILE = Path("dataset_metadata/catalog.json")
FOLDS = (
    "env_ubuntu22_template_v1",
    "env_ubuntu22_v1",
    "env_ubuntu22_v2",
    "single_host_ubuntu22_container",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values):
    return {
        "mean": sum(values) / len(values),
        "median": quantile(values, 0.5),
        "q1": quantile(values, 0.25),
        "q3": quantile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    masks = json.loads(MASK_FILE.read_text())
    catalog = json.loads(CATALOG_FILE.read_text())
    records = masks["masks"]

    environment = {}
    target = {}
    for sample in catalog["samples"]:
        environment[sample["sample_id"]] = sample["environment_group"]
        target[sample["sample_id"]] = sample.get("target")

    missing = sorted(set(records) - set(environment))
    if missing:
        raise AssertionError(f"{len(missing)} mask entries absent from the catalog")

    labelled = [key for key in records if target.get(key)]
    positives = [records[key]["positive_count"] for key in labelled]
    processes = [records[key]["graph_process_count"] for key in labelled]
    fractions = [p / f for p, f in zip(positives, processes)]
    largest = [key for key in labelled if records[key]["positive_count"] == max(positives)]
    largest_techniques = sorted({target[key] for key in largest})

    per_fold = {}
    for fold in FOLDS:
        keys = [key for key in labelled if environment[key] == fold]
        fold_fractions = [records[key]["positive_count"] /
                          records[key]["graph_process_count"] for key in keys]
        per_fold[fold] = {
            "graphs": len(keys),
            "mean_selected_fraction": sum(fold_fractions) / len(fold_fractions),
        }

    result = {
        "protocol": "coregate_lineage_target_statistics_v1",
        "mask_file": str(MASK_FILE),
        "mask_file_sha256": sha256(MASK_FILE),
        "catalog_file": str(CATALOG_FILE),
        "catalog_file_sha256": sha256(CATALOG_FILE),
        "labelled_graphs": len(labelled),
        "positive_nodes_per_graph": describe(positives),
        "process_nodes_per_graph": describe(processes),
        "largest_targets": {
            "positive_count": max(positives),
            "graph_count": len(largest),
            "techniques": largest_techniques,
        },
        "mean_selected_fraction": {
            "overall": sum(fractions) / len(fractions),
            **{fold: per_fold[fold]["mean_selected_fraction"] for fold in FOLDS},
            "min_fold": min(value["mean_selected_fraction"] for value in per_fold.values()),
            "max_fold": max(value["mean_selected_fraction"] for value in per_fold.values()),
        },
        "per_fold": per_fold,
        "label_invariance": {
            "changed_masks_after_label_permutation":
                masks["changed_mask_count_after_label_permutation"],
            "all_masks_byte_identical":
                masks["all_masks_byte_identical_after_label_permutation"],
        },
        "target_definition": masks["definition"],
        "allowed_input_fields": masks["allowed_input_fields"],
        "forbidden_input_fields": masks["forbidden_input_fields"],
    }

    sizes = result["positive_nodes_per_graph"]
    fractions_summary = result["mean_selected_fraction"]
    markdown = "\n".join([
        "# Execution-lineage target statistics",
        "",
        f"Source masks: `{MASK_FILE}` ({sha256(MASK_FILE)[:16]}...), "
        f"catalog: `{CATALOG_FILE}`.",
        f"{len(labelled)} labelled graphs; {masks['definition']}.",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Mean positive nodes | {sizes['mean']:.2f} |",
        f"| Median positive nodes | {sizes['median']:.0f} |",
        f"| Quartiles (Q1/Q3) | {sizes['q1']:.0f} / {sizes['q3']:.0f} |",
        f"| Range | {sizes['min']:.0f}-{sizes['max']:.0f} |",
        f"| Graphs at the maximum | {result['largest_targets']['graph_count']} "
        f"({', '.join(largest_techniques)}) |",
        f"| Mean selected fraction of process nodes | "
        f"{fractions_summary['overall'] * 100:.2f}% |",
        f"| Selected-fraction range across folds | "
        f"{fractions_summary['min_fold'] * 100:.2f}% - "
        f"{fractions_summary['max_fold'] * 100:.2f}% |",
        "",
        "The target is strongly skewed, and the largest targets are concentrated in",
        "one technique. The means above are descriptive: they do not show that",
        "target cardinality is class-independent.",
        "",
        "## Per fold",
        "",
        "| Fold | Graphs | Mean selected fraction |",
        "|---|---:|---:|",
        *[f"| {fold} | {per_fold[fold]['graphs']} | "
          f"{per_fold[fold]['mean_selected_fraction'] * 100:.2f}% |" for fold in FOLDS],
        "",
        "## Construction audit",
        "",
        f"- Allowed builder inputs: {', '.join(masks['allowed_input_fields'])}.",
        f"- Forbidden inputs: {', '.join(masks['forbidden_input_fields'])}.",
        f"- Masks changed by permuting all technique labels: "
        f"{masks['changed_mask_count_after_label_permutation']} of {len(records)}.",
    ])

    out = RESULT_DIR / "coregate_lineage_target_statistics.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    (RESULT_DIR / "coregate_lineage_target_statistics.md").write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
