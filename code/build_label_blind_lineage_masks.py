#!/usr/bin/env python3
"""Build and audit technique-label-blind execution-lineage masks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

from coregate.label_blind_lineage import (
    ALLOWED_ROW_FIELDS, FORBIDDEN_ROW_FIELDS, build_masks,
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path(
        "data/linux_telemetry/reference_candidates_template_v2/catalog.json"))
    parser.add_argument("--out", type=Path, default=Path(
        "data/experiment_results/leave_one_env_v1/label_blind_lineage_masks.json"))
    parser.add_argument("--audit-out", type=Path, default=Path(
        "data/experiment_results/leave_one_env_v1/label_blind_lineage_audit.json"))
    parser.add_argument("--permutation-seed", type=int, default=20260923)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text())
    rows = [row for row in catalog["samples"] if row.get("target")]
    original = build_masks(rows)

    permuted_rows = copy.deepcopy(rows)
    labels = [row["target"] for row in permuted_rows]
    random.Random(args.permutation_seed).shuffle(labels)
    for row, label in zip(permuted_rows, labels):
        row["target"] = label
    permuted = build_masks(permuted_rows)
    changed = [sample_id for sample_id in original
               if original[sample_id]["mask_sha256"] !=
               permuted[sample_id]["mask_sha256"]]
    counts = [row["positive_count"] for row in original.values()]
    output = {
        "protocol": "label_blind_execution_lineage_mask_v1",
        "definition": "process nodes associated with events whose collection-time execution_role is scenario",
        "catalog": str(args.catalog),
        "catalog_sha256": file_sha256(args.catalog),
        "builder_sha256": file_sha256(Path(__file__).parent / "coregate/label_blind_lineage.py"),
        "allowed_input_fields": list(ALLOWED_ROW_FIELDS),
        "forbidden_input_fields": list(FORBIDDEN_ROW_FIELDS),
        "samples": len(rows),
        "nonempty_masks": sum(row["positive_count"] > 0 for row in original.values()),
        "positive_count": {"min": min(counts), "median": sorted(counts)[len(counts)//2],
                           "max": max(counts)},
        "label_permutation_seed": args.permutation_seed,
        "changed_mask_count_after_label_permutation": len(changed),
        "all_masks_byte_identical_after_label_permutation": not changed,
        "changed_sample_ids": changed,
        "uses_target": False,
        "uses_technique_rules": False,
        "uses_control_kind": False,
        "uses_core_graph": False,
        "masks": original,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    audit = {key: value for key, value in output.items() if key != "masks"}
    args.audit_out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
