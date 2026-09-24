#!/usr/bin/env python3
"""Build a complete FullRule mask expanded with label-blind process context."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from coregate.label_blind_lineage import load_mask_records


def _ids(path):
    graph = json.loads(Path(path).read_text())
    return {
        str(node.get("id", index))
        for index, node in enumerate(graph.get("nodes", []))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path(
        "data/linux_telemetry/reference_candidates_template_v2/catalog.json"))
    parser.add_argument("--lineage-mask-file", type=Path, default=Path(
        "data/experiment_results/leave_one_env_v1/label_blind_lineage_masks.json"))
    parser.add_argument("--out", type=Path, default=Path(
        "data/experiment_results/leave_one_env_v1/fullrule_context_union_masks.json"))
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text())
    rows = [row for row in catalog["samples"] if row.get("target")]
    lineage, lineage_metadata = load_mask_records(args.lineage_mask_file)
    if not lineage_metadata.get("all_masks_byte_identical_after_label_permutation"):
        raise AssertionError("lineage component failed label-invariance audit")

    records = {}
    full_total = lineage_total = union_total = 0
    unchanged = 0
    for row in rows:
        sample_id = row["sample_id"]
        graph_ids = _ids(row["graph_path"])
        full_ids = _ids(row["core_graph_path"])
        lineage_ids = {
            str(value) for value in lineage[sample_id]["positive_node_ids"]
        }
        if not full_ids.issubset(graph_ids):
            raise AssertionError(f"FullRule nodes missing from graph: {sample_id}")
        if not lineage_ids.issubset(graph_ids):
            raise AssertionError(f"lineage nodes missing from graph: {sample_id}")
        union_ids = sorted(full_ids | lineage_ids)
        if set(union_ids) == full_ids:
            unchanged += 1
        payload = json.dumps(union_ids, separators=(",", ":")).encode()
        records[sample_id] = {
            "sample_id": sample_id,
            "positive_node_ids": union_ids,
            "positive_count": len(union_ids),
            "full_rule_positive_count": len(full_ids),
            "lineage_positive_count": len(lineage_ids),
            "added_context_count": len(set(union_ids) - full_ids),
            "mask_sha256": hashlib.sha256(payload).hexdigest(),
        }
        full_total += len(full_ids)
        lineage_total += len(lineage_ids)
        union_total += len(union_ids)

    output = {
        "protocol": "fullrule_context_union_mask_v1",
        "definition": "all FullRule nodes union all label-blind scenario-process lineage nodes",
        "catalog": str(args.catalog),
        "lineage_mask_file": str(args.lineage_mask_file),
        "samples": len(records),
        "full_rule_positive_nodes": full_total,
        "lineage_positive_nodes": lineage_total,
        "union_positive_nodes": union_total,
        "added_context_nodes": union_total - full_total,
        "unchanged_from_full_rule_samples": unchanged,
        "all_full_rule_nodes_retained": True,
        "all_lineage_nodes_retained": True,
        "uses_target": False,
        "uses_technique_rules": True,
        "uses_control_kind": True,
        "uses_core_graph": True,
        "lineage_component_label_invariant": True,
        "masks": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items()
                      if key != "masks"}, indent=2))


if __name__ == "__main__":
    main()
