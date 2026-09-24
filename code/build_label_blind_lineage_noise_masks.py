#!/usr/bin/env python3
"""Create deterministic positive-node dropout variants of lineage masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from coregate.label_blind_lineage import load_mask_records


def _score(seed, sample_id, node_id):
    payload = f"{seed}\0{sample_id}\0{node_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") / 2**256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--drop-rate", type=float, required=True)
    parser.add_argument("--noise-seed", type=int, default=20260921)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.drop_rate < 1:
        raise ValueError("drop rate must be in [0, 1)")

    records, metadata = load_mask_records(args.source)
    if not metadata.get("all_masks_byte_identical_after_label_permutation"):
        raise AssertionError("source masks failed label-invariance audit")

    output_records = {}
    original_total = retained_total = forced_nonempty = 0
    for sample_id, record in sorted(records.items()):
        original = [str(value) for value in record["positive_node_ids"]]
        scored = [(node_id, _score(args.noise_seed, sample_id, node_id))
                  for node_id in original]
        retained = [node_id for node_id, score in scored
                    if score >= args.drop_rate]
        if original and not retained:
            retained = [max(scored, key=lambda item: item[1])[0]]
            forced_nonempty += 1
        retained = sorted(retained)
        payload = json.dumps(retained, separators=(",", ":")).encode()
        changed = dict(record)
        changed["positive_node_ids"] = retained
        changed["positive_count"] = len(retained)
        changed["mask_sha256"] = hashlib.sha256(payload).hexdigest()
        output_records[sample_id] = changed
        original_total += len(original)
        retained_total += len(retained)

    output = {
        "protocol": "label_blind_execution_lineage_positive_dropout_v1",
        "source": str(args.source),
        "source_protocol": metadata.get("protocol"),
        "drop_rate_requested": args.drop_rate,
        "noise_seed": args.noise_seed,
        "samples": len(output_records),
        "original_positive_nodes": original_total,
        "retained_positive_nodes": retained_total,
        "dropped_positive_nodes": original_total - retained_total,
        "drop_rate_realized": ((original_total - retained_total) /
                               max(1, original_total)),
        "forced_nonempty_masks": forced_nonempty,
        "nested_hash_threshold": True,
        "all_masks_byte_identical_after_label_permutation": True,
        "changed_mask_count_after_label_permutation": 0,
        "uses_target": False,
        "uses_technique_rules": False,
        "uses_control_kind": False,
        "uses_core_graph": False,
        "masks": output_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items()
                      if key != "masks"}, indent=2))


if __name__ == "__main__":
    main()
