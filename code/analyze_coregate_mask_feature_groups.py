#!/usr/bin/env python3
"""Localize which simple weak-mask statistics drive the P0 audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from run_coregate_mask_only_audit import SEEDS, _graph_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path(
        "data/linux_telemetry/reference_candidates_template_v2/catalog.json"))
    parser.add_argument("--out", type=Path, default=Path(
        "data/experiment_results/leave_one_env_v1/coregate_mask_feature_group_audit.json"))
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    rows = [row for row in catalog["samples"] if row.get("target")]
    environments = sorted({row.get("environment_group") for row in rows})
    operations = sorted({str(edge.get("operation", "unknown"))
                         for row in rows
                         for edge in json.loads(Path(row["graph_path"]).read_text()).get("edges", [])})
    values = [_graph_features(row, operations) for row in rows]
    names = sorted(values[0])
    matrix = np.asarray([[v[n] for n in names] for v in values], dtype=float)
    labels = np.asarray([row["target"] for row in rows])
    envs = np.asarray([row.get("environment_group") for row in rows])
    groups = {
        "mask_cardinality_only": [i for i, n in enumerate(names) if n in {"num_nodes", "num_edges", "num_positive_mask_nodes", "positive_ratio", "num_positive_edges", "positive_edge_ratio"}],
        "node_type_counts": [i for i, n in enumerate(names) if n.startswith("num_") or n.startswith("positive_") if "edge" not in n],
        "positive_node_mask_types": [i for i, n in enumerate(names) if n.startswith("positive_") and "edge" not in n],
        "positive_edge_operations": [i for i, n in enumerate(names) if n.startswith("positive_edge_operation_")],
        "all": list(range(len(names))),
    }
    output = {"protocol": "coregate_mask_feature_group_audit_v1", "classes": 45,
              "groups": {}, "folds": environments, "seeds": list(SEEDS)}
    for group, indices in groups.items():
        fold_metrics = []
        for seed in SEEDS:
            for heldout in environments:
                train = envs != heldout
                test = ~train
                scaler = StandardScaler()
                train_x = scaler.fit_transform(matrix[train][:, indices])
                test_x = scaler.transform(matrix[test][:, indices])
                model = LogisticRegression(max_iter=2000, class_weight="balanced",
                                           multi_class="multinomial", random_state=seed)
                model.fit(train_x, labels[train])
                pred = model.predict(test_x)
                fold_metrics.append({"seed": seed, "heldout": heldout,
                                     "accuracy": float(accuracy_score(labels[test], pred)),
                                     "macro_f1": float(f1_score(labels[test], pred, average="macro", zero_division=0))})
        seed_acc = [np.mean([r["accuracy"] for r in fold_metrics if r["seed"] == s]) for s in SEEDS]
        seed_f1 = [np.mean([r["macro_f1"] for r in fold_metrics if r["seed"] == s]) for s in SEEDS]
        output["groups"][group] = {"feature_count": len(indices),
            "features": [names[i] for i in indices],
            "accuracy": {"mean": float(np.mean(seed_acc)), "std_ddof1": float(np.std(seed_acc, ddof=1))},
            "macro_f1": {"mean": float(np.mean(seed_f1)), "std_ddof1": float(np.std(seed_f1, ddof=1))},
            "folds": fold_metrics}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: {m: v[m] for m in ("accuracy", "macro_f1")} for k, v in output["groups"].items()}, indent=2))


if __name__ == "__main__":
    main()
