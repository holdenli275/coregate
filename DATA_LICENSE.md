# Dataset license

The CoreGate Linux telemetry candidate set (`coregate_dataset_template_v2.tar.zst`,
extracted to `data/linux_telemetry/reference_candidates_template_v2/`) and the
released result artifacts in `results/` and `dataset_metadata/` are licensed
under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

The dataset archive itself is released upon acceptance of the accompanying
paper; this repository ships the catalog metadata, the data card, and the
per-sample checksums.

- Human-readable summary: https://creativecommons.org/licenses/by/4.0/
- Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode

## Attribution

When you use or redistribute the dataset, attribute it as:

> Anonymous Authors (double-blind review copy)
> (2026). *CoreGate Linux Telemetry Candidate Set (template v2)* [Data set].
> Zenodo. https://doi.org/10.5281/zenodo.YYYYYYY

and keep the following notice with any redistributed copy:

> This dataset is released under CC BY 4.0. It was produced by executing Atomic
> Red Team tests in a controlled Ubuntu 22.04 lab. Upstream projects retain
> their own licenses.

## Scope

CC BY 4.0 covers the release's own contributions: the capture protocol, the
derived graph and evidence-mask representations, the catalog, and the packaging.

It does not relicense third-party material. Atomic Red Team content remains
under its upstream license and is referenced here only as provenance. MITRE
ATT&CK technique identifiers are used under the MITRE ATT&CK terms of use. See
`docs/DATA_PROVENANCE_AND_PRIVACY.md`.
