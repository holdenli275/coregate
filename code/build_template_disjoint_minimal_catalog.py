#!/usr/bin/env python3
"""Freeze a deterministic 45-way template-disjoint LOEO catalog.

One template group per technique is held out. Since every group in the
current attack catalog belongs to exactly one technique and every technique
has at least two groups, this keeps all 45 labels in both train and test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path(
        "data/linux_telemetry/reference_candidates_template_v2/catalog.json"))
    parser.add_argument("--out", type=Path, default=Path(
        "data/experiment_results/leave_one_env_v1/template_disjoint_minimal_catalog.json"))
    parser.add_argument("--audit-out", type=Path, default=Path(
        "data/experiment_results/leave_one_env_v1/template_disjoint_minimal_split.json"))
    args = parser.parse_args()
    source = json.loads(args.catalog.read_text())
    attack = [row for row in source["samples"] if row.get("target")]
    by_class = defaultdict(list)
    by_group = defaultdict(list)
    for row in attack:
        by_class[row["target"]].append(row)
        by_group[row["template_group"]].append(row)
    if any(len({row["template_group"] for row in rows}) < 2
           for rows in by_class.values()):
        raise AssertionError("every class needs at least two template groups")
    if any(len({row["target"] for row in rows}) != 1
           for rows in by_group.values()):
        raise AssertionError("a template group spans multiple labels")
    heldout_groups = sorted(
        min({row["template_group"] for row in rows})
        for rows in by_class.values())
    heldout = set(heldout_groups)
    rows = []
    for row in source["samples"]:
        copy = dict(row)
        if row.get("target"):
            copy["original_environment_group"] = row.get("environment_group")
            copy["environment_group"] = (
                "template_disjoint_test" if row["template_group"] in heldout
                else "template_disjoint_train")
            rows.append(copy)
        # Keep controls out of this 45-way closed-set diagnostic.
    train = [row for row in rows if row["environment_group"] == "template_disjoint_train"]
    test = [row for row in rows if row["environment_group"] == "template_disjoint_test"]
    train_classes = {row["target"] for row in train}
    test_classes = {row["target"] for row in test}
    overlap = {row["template_group"] for row in train} & {
        row["template_group"] for row in test}
    if len(train_classes) != 45 or len(test_classes) != 45 or overlap:
        raise AssertionError("template-disjoint split coverage/overlap check failed")
    payload = "\n".join(sorted(row["sample_id"] for row in rows)).encode()
    audit = {
        "protocol": "coregate_template_disjoint_minimal_v1",
        "source_catalog": str(args.catalog), "rows": len(rows), "classes": 45,
        "heldout_template_groups": heldout_groups,
        "train_samples": len(train), "test_samples": len(test),
        "train_class_count": len(train_classes),
        "test_class_count": len(test_classes),
        "train_template_group_count": len({r["template_group"] for r in train}),
        "test_template_group_count": len({r["template_group"] for r in test}),
        "template_group_overlap_count": len(overlap),
        "catalog_row_hash": hashlib.sha256(payload).hexdigest()[:16],
        "sanity_checks": {"template_overlap_zero": True,
                          "train_class_coverage_45": True,
                          "test_class_coverage_45": True},
    }
    output = dict(source)
    output["schema"] = "linux-atomic-template-disjoint-eval-v1"
    output["samples"] = rows
    output["template_disjoint_split"] = audit
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    args.audit_out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
