#!/usr/bin/env python3
"""Leave-one-environment-out evaluation for the CoreGate pipeline.

The file name keeps the internal protocol token recorded in the run JSONs as
``training_protocol: episode_tgn``.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import run_coregate_tgn_episode as episode
import graph_base as base
from coregate.tgn_episode import embed_episode_graphs, train_episode_tgn
from coregate.label_blind_lineage import load_mask_records


def _attach_predicted_core_scores(train_data, test_data):
    """Fit a train-only node gate; test core masks are never read."""
    values = np.concatenate([data.x.detach().cpu().numpy() for data in train_data])
    labels = np.concatenate([
        data.core_node_mask.detach().cpu().numpy().astype(np.int64)
        for data in train_data
    ])
    if len(np.unique(labels)) < 2:
        scores = [np.zeros(data.x.size(0), dtype=np.float32)
                  for data in train_data + test_data]
    else:
        fitted = LogisticRegression(max_iter=1000, class_weight="balanced",
                                    random_state=0).fit(values, labels)
        scores = [fitted.predict_proba(data.x.detach().cpu().numpy())[:, 1]
                  for data in train_data + test_data]
    for data, score in zip(train_data + test_data, scores):
        data.core_score = torch.as_tensor(score, dtype=torch.float32)


def _set_train_core_mask_source(train_data, source, seed):
    """Select the training-only evidence supervision used by CoreGate."""
    if source == "evidence":
        return
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 99173)
    for data in train_data:
        count = data.x.size(0)
        if source == "none":
            mask = torch.zeros(count, dtype=torch.bool)
        elif source == "all":
            mask = torch.ones(count, dtype=torch.bool)
        elif source in {"random", "lineage_random"}:
            if source == "lineage_random":
                reference = getattr(data, "lineage_node_mask", None)
                if reference is None or reference.numel() != count:
                    raise AssertionError("lineage_random requires a lineage mask")
            else:
                reference = getattr(data, "core_node_mask", None)
            positives = int(reference.sum()) if reference is not None else max(1, count // 4)
            positives = min(max(1, positives), count)
            permutation = torch.randperm(count, generator=generator)
            mask = torch.zeros(count, dtype=torch.bool)
            mask[permutation[:positives]] = True
        elif source == "lineage":
            reference = getattr(data, "lineage_node_mask", None)
            if reference is None or reference.numel() != count:
                raise AssertionError("lineage supervision requires a lineage mask")
            mask = reference.clone().bool()
        elif source == "context_union":
            reference = getattr(data, "context_union_node_mask", None)
            if reference is None or reference.numel() != count:
                raise AssertionError("context_union requires a precomputed mask")
            mask = reference.clone().bool()
        elif source == "process_only":
            reference = getattr(data, "core_node_mask", None)
            if reference is None or reference.numel() != count:
                raise AssertionError("process_only requires the serialized evidence mask")
            node_types = getattr(data, "node_type", None)
            if node_types is None or node_types.numel() != count:
                # encoder_input stores process as type 0 in the final one-hot block.
                node_types = data.x[:, 12:16].argmax(dim=1)
            mask = reference.clone().bool() & (node_types.to(reference.device) == 0)
        else:
            raise ValueError(source)
        data.core_node_mask = mask


def _hash_ids(values):
    payload = "\n".join(str(value) for value in values).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _shuffle_edge_times(data, sample_id):
    """Break edge-to-rank alignment while preserving all graph tensors."""
    if data.edge_time.numel() < 2:
        return
    generator = torch.Generator()
    generator.manual_seed(int(hashlib.sha256(
        str(sample_id).encode()).hexdigest()[:16], 16))
    permutation = torch.randperm(data.edge_time.numel(), generator=generator)
    data.edge_time = data.edge_time[permutation]


def _ordered_catalog_rows(catalog):
    """Return rows in exactly the order used by load_rows_with_environment."""
    by_class = {}
    for row in catalog["samples"]:
        if row.get("target"):
            by_class.setdefault(row["target"], []).append(row)
    rows = []
    for label in sorted(by_class):
        rows.extend(sorted(by_class[label], key=lambda row: row["sample_id"]))
    return rows


def _precomputed_mask_tensor(graph, record, source):
    positive = {str(value) for value in record["positive_node_ids"]}
    node_ids = [str(value) for value in graph.get("node_ids", [])]
    if len(node_ids) != graph["x"].size(0):
        raise AssertionError(f"{source} mask node IDs do not match graph tensor")
    mask = torch.tensor([node_id in positive for node_id in node_ids],
                        dtype=torch.bool)
    if int(mask.sum()) != int(record["positive_count"]):
        raise AssertionError(f"serialized {source} positive count does not match graph")
    return mask


def _split_audit_metadata(catalog, holdout, support_per_class=1,
                          excluded=None):
    """Reconstruct deterministic sample/support IDs used by the LOEO split."""
    excluded = set(excluded or ())
    by_class = {}
    for row in catalog["samples"]:
        label = row.get("target")
        if label and label not in excluded:
            by_class.setdefault(label, []).append(row)
    rows = []
    for label in sorted(by_class):
        values = sorted(by_class[label], key=lambda row: row["sample_id"])
        test_count = 1 if len(values) == 2 else min(2, len(values) - 1)
        rows.extend((row, label, index >= len(values) - test_count)
                    for index, row in enumerate(values))
    # The runner's training split is by environment, while the episode support
    # rule takes the first support_per_class item in each sorted class stream.
    # Include all rows in the held-out environment; the marker is only the
    # within-class support/query split used by episodic optimization.
    train_rows = [row for row, _, _ in rows
                  if row.get("environment_group") != holdout]
    test_rows = [row for row, _, _ in rows
                 if row.get("environment_group") == holdout]
    support_ids = []
    seen = {}
    for row, label, _ in rows:
        if row.get("environment_group") != holdout:
            count = seen.get(label, 0)
            if count < support_per_class:
                support_ids.append(row["sample_id"])
                seen[label] = count + 1
    return {
        "train_graph_hash": _hash_ids([row["sample_id"] for row in train_rows]),
        "test_graph_hash": _hash_ids([row["sample_id"] for row in test_rows]),
        "support_graph_hash": _hash_ids(support_ids),
        "train_graph_count": len(train_rows),
        "test_graph_count": len(test_rows),
        "support_graph_count": len(support_ids),
    }


def run_environment(args, catalog, holdout, seed):
    catalog["_catalog_path"] = args.catalog
    (graphs, labels, environment_mask, classes, environment_ids,
     environment_names) = episode.load_rows_with_environment(catalog, args.graph_field)
    ordered_rows = _ordered_catalog_rows(catalog)
    if len(ordered_rows) != len(graphs):
        raise AssertionError("catalog row order does not match loaded graphs")
    excluded = set(args.exclude_class)
    if excluded:
        keep = np.asarray([label not in excluded for label in labels], dtype=bool)
        graphs = [graph for graph, flag in zip(graphs, keep) if flag]
        labels = labels[keep]
        environment_ids = environment_ids[keep]
        environment_mask = environment_mask[keep]
        ordered_rows = [row for row, flag in zip(ordered_rows, keep) if flag]
        classes = sorted(set(labels.tolist()))
    class_to_id = {label: index for index, label in enumerate(classes)}
    ids = np.asarray([class_to_id[label] for label in labels])
    target_mask = np.asarray([label is not None for label in labels])
    holdout_id = environment_names.index(holdout)
    train_mask = (environment_ids != holdout_id) & target_mask
    test_mask = (environment_ids == holdout_id) & target_mask
    heldout_templates = {row.get("template_group") for row, is_test in
                         zip(ordered_rows, test_mask) if is_test}
    if args.joint_template_disjoint:
        if None in heldout_templates:
            raise AssertionError("joint holdout requires template groups")
        train_mask &= np.asarray([
            row.get("template_group") not in heldout_templates
            for row in ordered_rows], dtype=bool)
        if set(labels[train_mask]) != set(labels):
            raise AssertionError(f"joint holdout loses training classes: {holdout}")
    train_graphs = [graph for graph, keep in zip(graphs, train_mask) if keep]
    test_graphs = [graph for graph, keep in zip(graphs, test_mask) if keep]
    train_rows = [row for row, keep in zip(ordered_rows, train_mask) if keep]
    test_rows = [row for row, keep in zip(ordered_rows, test_mask) if keep]
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]
    train_ids = ids[train_mask].tolist()
    train_environments = environment_ids[train_mask].tolist()
    test_ids = ids[test_mask].tolist()
    train_data = [base.encoder_input(graph, label)
                  for graph, label in zip(train_graphs, train_ids)]
    test_data = [base.encoder_input(graph, label)
                 for graph, label in zip(test_graphs, test_ids)]
    if args.edge_time_mode == "shuffled":
        for data, row in zip(train_data + test_data, train_rows + test_rows):
            _shuffle_edge_times(data, row["sample_id"])
    elif args.edge_time_mode == "constant":
        for data in train_data + test_data:
            data.edge_time.fill_(0.5)
    train_templates = {row.get("template_group") for row in train_rows}
    test_templates = {row.get("template_group") for row in test_rows}
    template_overlap = train_templates & test_templates
    if args.joint_template_disjoint and template_overlap:
        raise AssertionError("joint holdout retained a test template in training")
    lineage_records = None
    lineage_metadata = None
    if args.train_core_mask in {"lineage", "lineage_random"}:
        lineage_records, lineage_metadata = load_mask_records(args.lineage_mask_file)
        for data, graph, row in zip(train_data, train_graphs, train_rows):
            record = lineage_records.get(row["sample_id"])
            if record is None:
                raise AssertionError(f"missing lineage mask for {row['sample_id']}")
            data.lineage_node_mask = _precomputed_mask_tensor(
                graph, record, "lineage")
    context_union_records = None
    context_union_metadata = None
    if args.train_core_mask == "context_union":
        context_union_records, context_union_metadata = load_mask_records(
            args.context_union_mask_file)
        if not context_union_metadata.get("all_full_rule_nodes_retained"):
            raise AssertionError("context union does not retain FullRule nodes")
        for data, graph, row in zip(train_data, train_graphs, train_rows):
            record = context_union_records.get(row["sample_id"])
            if record is None:
                raise AssertionError(
                    f"missing context-union mask for {row['sample_id']}")
            data.context_union_node_mask = _precomputed_mask_tensor(
                graph, record, "context_union")
    reference_core_counts = [int(data.core_node_mask.sum())
                             for data in train_data]
    reference_lineage_counts = [
        int(getattr(data, "lineage_node_mask", torch.zeros(0)).sum())
        for data in train_data
    ]
    reference_context_union_counts = [
        int(getattr(data, "context_union_node_mask", torch.zeros(0)).sum())
        for data in train_data
    ]
    _set_train_core_mask_source(train_data, args.train_core_mask, seed)
    train_core_counts = [int(data.core_node_mask.sum()) for data in train_data]
    max_core_count_delta = max(
        (abs(after - before) for before, after in
         zip(reference_core_counts, train_core_counts)), default=0)
    if args.train_core_mask == "random" and max_core_count_delta != 0:
        raise AssertionError("shuffled evidence changed per-graph positive count")
    lineage_count_delta = max(
        (abs(after - before) for before, after in
         zip(reference_lineage_counts, train_core_counts)), default=0)
    if args.train_core_mask in {"lineage", "lineage_random"} and lineage_count_delta != 0:
        raise AssertionError("lineage mask source changed per-graph positive count")
    context_union_count_delta = max(
        (abs(after - before) for before, after in
         zip(reference_context_union_counts, train_core_counts)), default=0)
    if args.train_core_mask == "context_union" and context_union_count_delta != 0:
        raise AssertionError("context-union positive count changed")
    if args.pooling == "predicted_core":
        _attach_predicted_core_scores(train_data, test_data)
    if args.test_core_mask == "none":
        for data in test_data:
            data.core_node_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
            data.core_edge_mask = torch.zeros(data.edge_time.numel(), dtype=torch.bool)
    test_node_mask_sum = int(sum(int(data.core_node_mask.sum()) for data in test_data))
    test_edge_mask_sum = int(sum(int(data.core_edge_mask.sum()) for data in test_data))
    if args.test_core_mask == "none" and (test_node_mask_sum or test_edge_mask_sum):
        raise AssertionError("held-out evidence mask was not fully cleared")
    # Fix initialization before constructing the encoder.  Training functions also
    # reseed internally, but that was too late to make process-level repeats
    # deterministic.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = episode.new_model(len(classes))
    last_loss, last_parts, support, query = train_episode_tgn(
        model, train_data, train_ids, args.epochs, seed,
        support_per_class=args.support_per_class,
        metric_weight=args.metric_weight, margin_weight=args.margin_weight,
        temperature=args.temperature, margin=args.margin,
        core_pool_weight=args.core_pool_weight,
        pooling=args.pooling,
        core_gate_weight=args.core_gate_weight,
        share_gate_forward=args.share_gate_forward,
    )
    training_protocol = "episode_tgn"
    train_embeddings = embed_episode_graphs(
        model, train_data, args.core_pool_weight, args.pooling)
    test_embeddings = embed_episode_graphs(
        model, test_data, args.core_pool_weight, args.pooling)
    # Downstream classifier control: fit the same LogisticRegression protocol
    # on graph embeddings alone.  This is computed from the same trained
    # encoder checkpoint as the embedding+prototype baseline below, so it
    # isolates the contribution of appended support similarities without
    # changing the encoder, split, or training budget.
    scaler_class = StandardScaler if args.classifier_scaler == "standard" else MinMaxScaler
    embedding_scaler = (scaler_class() if args.classifier_scaler == "standard"
                        else scaler_class(feature_range=(-1.0, 1.0)))
    embedding_train_features = embedding_scaler.fit_transform(
        train_embeddings.numpy())
    embedding_test_features = embedding_scaler.transform(
        test_embeddings.numpy())
    embedding_only_classifier = episode.make_classifier(
        args.classifier, seed, args.classifier_C).fit(
            embedding_train_features, train_labels)
    embedding_only_metrics = episode.metrics(
        embedding_only_classifier.predict(embedding_test_features), test_labels)
    train_sim, test_sim = episode.support_similarity(
        train_embeddings, test_embeddings, train_ids,
        (episode.all_class_support(train_ids)
         if args.similarity_prototype == "all" else support))
    train_raw = np.concatenate([train_embeddings.numpy(), train_sim], axis=1)
    test_raw = np.concatenate([test_embeddings.numpy(), test_sim], axis=1)
    scaler = (scaler_class() if args.classifier_scaler == "standard"
              else scaler_class(feature_range=(-1.0, 1.0)))
    train_features = scaler.fit_transform(train_raw)
    test_features = scaler.transform(test_raw)
    baseline = episode.make_classifier(args.classifier, seed, args.classifier_C).fit(
        train_features, train_labels)
    baseline_metrics = episode.metrics(baseline.predict(test_features), test_labels)
    # Only ``--augmentation none`` is supported: the flag and this record are
    # kept because every released command writes them.
    augmentation = {"mode": "none", "synthetic_samples": 0}
    if args.joint_template_disjoint:
        split_audit = {
            "train_graph_hash": _hash_ids([row["sample_id"] for row in train_rows]),
            "test_graph_hash": _hash_ids([row["sample_id"] for row in test_rows]),
            "support_graph_hash": _hash_ids([
                train_rows[index]["sample_id"]
                for label in sorted(support) for index in support[label]]),
            "train_graph_count": len(train_rows),
            "test_graph_count": len(test_rows),
            "support_graph_count": sum(len(indices) for indices in support.values()),
        }
    else:
        split_audit = _split_audit_metadata(
            catalog, holdout, args.support_per_class, args.exclude_class)
    return {
        "holdout_environment": holdout,
        "seed": seed,
        "classes": len(classes),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "test_class_count": len(set(test_labels.tolist())),
        "joint_template_disjoint": args.joint_template_disjoint,
        "train_template_groups": len(train_templates),
        "test_template_groups": len(test_templates),
        "template_group_overlap_count": len(template_overlap),
        "edge_time_mode": args.edge_time_mode,
        "training_protocol": training_protocol,
        "core_pool_weight": args.core_pool_weight,
        "core_gate_weight": args.core_gate_weight,
        "pooling": args.pooling,
        "test_core_mask": args.test_core_mask,
        "train_core_mask": args.train_core_mask,
        "episode_query_samples": len(query),
        "similarity_prototype": args.similarity_prototype,
        "classifier_scaler": args.classifier_scaler,
        "share_gate_forward": args.share_gate_forward,
        "last_loss": last_loss,
        "last_loss_parts": last_parts,
        "baseline": baseline_metrics,
        "embedding_only": embedding_only_metrics,
        "augmentation": augmentation,
        "trainable_parameters": int(sum(parameter.numel()
                                         for parameter in model.parameters()
                                         if parameter.requires_grad)),
        "reference_train_core_positive_nodes": int(sum(reference_core_counts)),
        "reference_train_lineage_positive_nodes": int(sum(reference_lineage_counts)),
        "train_core_positive_nodes": int(sum(train_core_counts)),
        "train_core_positive_count_max_abs_delta": int(max_core_count_delta),
        "train_lineage_positive_count_max_abs_delta": int(lineage_count_delta),
        "lineage_mask_file": (str(args.lineage_mask_file)
                              if lineage_metadata is not None else None),
        "lineage_label_invariance_check": (
            bool(lineage_metadata.get("all_masks_byte_identical_after_label_permutation"))
            if lineage_metadata is not None else None),
        "context_union_mask_file": (str(args.context_union_mask_file)
                                    if context_union_metadata is not None else None),
        "context_union_retains_full_rule": (
            bool(context_union_metadata.get("all_full_rule_nodes_retained"))
            if context_union_metadata is not None else None),
        "train_context_union_positive_count_max_abs_delta": int(
            context_union_count_delta),
        "test_core_node_mask_sum_after_clear": test_node_mask_sum,
        "test_core_edge_mask_sum_after_clear": test_edge_mask_sum,
        "test_mask_leakage_check": bool(
            args.test_core_mask == "none" and
            test_node_mask_sum == 0 and test_edge_mask_sum == 0),
        **split_audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path,
                        default=Path("data/linux_telemetry/reference_candidates_template_v2/catalog.json"))
    parser.add_argument("--graph-field", choices=("graph_path", "core_graph_path"), default="core_graph_path")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--support-per-class", type=int, default=1)
    parser.add_argument("--similarity-prototype", choices=("support", "all"),
                        default="support",
                        help="support-only or all-training class prototypes")
    parser.add_argument("--metric-weight", type=float, default=0.50)
    parser.add_argument("--margin-weight", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--augmentation", choices=("none",), default="none",
                        help="augmentation backends are not part of this release")
    parser.add_argument("--classifier-C", type=float, default=1.0)
    parser.add_argument("--classifier", choices=("logreg", "svm", "lda"), default="logreg")
    parser.add_argument("--classifier-scaler", choices=("minmax", "standard"),
                        default="minmax",
                        help="downstream feature scaler; minmax preserves the locked main protocol")
    parser.add_argument("--share-gate-forward", action="store_true",
                        help="reuse the training forward's node embeddings for gate BCE; supplementary speed path")
    parser.add_argument("--core-pool-weight", type=float, default=0.0,
                        help="evidence-core pooling multiplier; 0 keeps mean pooling")
    parser.add_argument("--pooling", choices=("ordinary", "core", "norm_attention", "logit_attention", "predicted_core", "learned_core", "sagpool"),
                        default="ordinary",
                        help="graph pooling rule; learned_core trains an evidence gate on train masks only; sagpool uses a fixed 0.5 retention ratio")
    parser.add_argument("--core-gate-weight", type=float, default=0.0,
                        help="training-only BCE weight for the learned evidence gate")
    parser.add_argument("--test-core-mask", choices=("oracle", "none"), default="oracle",
                        help="use serialized evidence mask at test time or audit without it")
    parser.add_argument("--train-core-mask", choices=("evidence", "process_only", "lineage", "lineage_random", "context_union", "random", "none", "all"),
                        default="evidence",
                        help="training-only gate supervision source for learned_core")
    parser.add_argument("--lineage-mask-file", type=Path, default=Path(
                        "data/experiment_results/leave_one_env_v1/label_blind_lineage_masks.json"),
                        help="precomputed label-blind execution-lineage masks")
    parser.add_argument("--context-union-mask-file", type=Path, default=Path(
                        "data/experiment_results/leave_one_env_v1/fullrule_context_union_masks.json"),
                        help="precomputed FullRule union label-blind context masks")
    parser.add_argument("--exclude-class", action="append", default=[],
                        help="exclude a technique label from this run; repeat for multiple labels")
    parser.add_argument("--holdout", default=None,
                        help="run only one named environment (useful for quick ablations)")
    parser.add_argument("--joint-template-disjoint", action="store_true",
                        help="also remove held-out template groups from training")
    parser.add_argument("--edge-time-mode", choices=("ordered", "shuffled", "constant"),
                        default="ordered",
                        help="per-graph edge-rank control; shuffled preserves the time multiset")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    environments = sorted({row.get("environment_group") for row in catalog["samples"]
                           if row.get("target")})
    if args.holdout is not None:
        if args.holdout not in environments:
            raise ValueError(f"unknown holdout environment: {args.holdout}")
        environments = [args.holdout]
    results = [run_environment(args, catalog, environment, args.seed + index)
               for index, environment in enumerate(environments)]
    output = {
        "protocol": "coregate_tgn_leave_one_environment_out_v1",
        "catalog": str(args.catalog),
        "view": "Core_Graph" if args.graph_field == "core_graph_path" else "Raw_Graph",
        "exclude_classes": args.exclude_class,
        "environments": environments,
        "results": results,
        "mean_baseline": {
            key: float(np.mean([result["baseline"][key] for result in results]))
            for key in results[0]["baseline"]
        },
        "limitations": [
            "Environment groups are isolated container batches on one host, not independent hosts.",
            "Core graphs use automatic evidence boundaries and have no independent manual gold.",
            "Targets are automatic weak supervision, not independently validated attack evidence.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in
                      ("protocol", "environments", "mean_baseline")}, indent=2))


if __name__ == "__main__":
    main()
