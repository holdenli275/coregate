"""Technique-label-blind execution-lineage masks for CoreGate training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ALLOWED_ROW_FIELDS = ("sample_id", "annotation_path", "events_path", "graph_path")
FORBIDDEN_ROW_FIELDS = (
    "target", "technique", "guid", "template_group", "supplemental_trigger",
    "core_graph_path", "source_manifest",
)


def mask_input(row):
    """Return the complete, auditable input surface for mask construction."""
    missing = [name for name in ALLOWED_ROW_FIELDS if not row.get(name)]
    if missing:
        raise ValueError(f"lineage mask input is missing fields: {missing}")
    return {name: str(row[name]) for name in ALLOWED_ROW_FIELDS}


def _jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_mask_record(inputs):
    """Build a process-only foreground mask without reading a class label."""
    if set(inputs) != set(ALLOWED_ROW_FIELDS):
        raise ValueError("lineage builder received fields outside its allowlist")
    scenario_events = {
        str(row["event_id"])
        for row in _jsonl(inputs["annotation_path"])
        if row.get("execution_role") == "scenario"
    }
    positive_ids = {
        str(row["process"])
        for row in _jsonl(inputs["events_path"])
        if str(row.get("event_id", "")) in scenario_events and
        row.get("process") is not None
    }
    graph = json.loads(Path(inputs["graph_path"]).read_text())
    process_ids = {
        str(node.get("id", index))
        for index, node in enumerate(graph.get("nodes", []))
        if node.get("type") == "process"
    }
    positive_ids &= process_ids
    if not positive_ids:
        raise ValueError(f"empty scenario-process mask for {inputs['sample_id']}")
    values = sorted(positive_ids)
    payload = json.dumps(values, separators=(",", ":")).encode()
    return {
        "sample_id": inputs["sample_id"],
        "positive_node_ids": values,
        "positive_count": len(values),
        "mask_sha256": hashlib.sha256(payload).hexdigest(),
        "scenario_event_count": len(scenario_events),
        "graph_process_count": len(process_ids),
    }


def build_masks(rows):
    records = [build_mask_record(mask_input(row)) for row in rows]
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("duplicate sample IDs in lineage mask inputs")
    return {row["sample_id"]: row for row in records}


def load_mask_records(path):
    value = json.loads(Path(path).read_text())
    records = value.get("masks", value)
    if isinstance(records, list):
        records = {row["sample_id"]: row for row in records}
    return records, value
