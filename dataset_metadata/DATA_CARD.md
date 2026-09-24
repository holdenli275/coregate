# Template augmented Linux telemetry candidate set

This supplement contains 360 attack candidates across 45 ATT&CK sub-techniques and 14 controls. It adds one independently captured `variant_b` execution template per reviewed Atomic contract and one SSH environment batch. The original v0.1 snapshot is unchanged.

Every added run passed collector counter balance, zero ring-buffer drops, namespace checks, semantic postconditions, and automatic core-graph checks. Template groups are GUID plus variant; environment groups are retained for leave-one-environment-out evaluation. Independent manual gold remains zero, so this is a research candidate set rather than a formal benchmark.

The new environment covers all 45 classes. Thirty-seven classes now have two template groups and eight classes have more than two because the original catalog already contained multiple Atomic GUIDs.
