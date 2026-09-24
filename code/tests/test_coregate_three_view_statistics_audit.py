from run_coregate_three_view_statistics_audit import statistic_features


def test_raw_and_mask_statistics_use_disjoint_inputs():
    graph = {
        "nodes": [{"id": 1, "type": "process"},
                  {"id": 2, "type": "file"},
                  {"id": 3, "type": "process"}],
        "edges": [{"source": 1, "target": 2, "operation": "write"},
                  {"source": 3, "target": 2, "operation": "read"}],
    }
    types = ["file", "process"]
    operations = ["read", "write"]
    raw, masked = statistic_features(graph, [1, 2], types, operations)
    assert raw["raw_num_nodes"] == 3
    assert raw["raw_edge_operation:read"] == 1
    assert masked["mask_num_nodes"] == 2
    assert masked["mask_num_edges"] == 1
    assert masked["mask_edge_operation:read"] == 0

    expanded = {**graph, "edges": graph["edges"] + [
        {"source": 3, "target": 3, "operation": "read"}]}
    expanded_raw, expanded_mask = statistic_features(
        expanded, [1, 2], types, operations)
    assert expanded_raw != raw
    assert expanded_mask == masked

    same_raw, smaller_mask = statistic_features(graph, [1], types, operations)
    assert same_raw == raw
    assert smaller_mask != masked
