# Execution-lineage target statistics

Source masks: `results/label_blind_lineage_masks.json` (93cc44e40103c0bb...), catalog: `dataset_metadata/catalog.json`.
360 labelled graphs; process nodes associated with events whose collection-time execution_role is scenario.

| Statistic | Value |
|---|---:|
| Mean positive nodes | 21.54 |
| Median positive nodes | 4 |
| Quartiles (Q1/Q3) | 2 / 5 |
| Range | 1-879 |
| Graphs at the maximum | 7 (T1553.004) |
| Mean selected fraction of process nodes | 21.27% |
| Selected-fraction range across folds | 21.07% - 21.70% |

The target is strongly skewed, and the largest targets are concentrated in
one technique. The means above are descriptive: they do not show that
target cardinality is class-independent.

## Per fold

| Fold | Graphs | Mean selected fraction |
|---|---:|---:|
| env_ubuntu22_template_v1 | 55 | 21.18% |
| env_ubuntu22_v1 | 54 | 21.07% |
| env_ubuntu22_v2 | 163 | 21.14% |
| single_host_ubuntu22_container | 88 | 21.70% |

## Construction audit

- Allowed builder inputs: sample_id, annotation_path, events_path, graph_path.
- Forbidden inputs: target, technique, guid, template_group, supplemental_trigger, core_graph_path, source_manifest.
- Masks changed by permuting all technique labels: 0 of 360.
