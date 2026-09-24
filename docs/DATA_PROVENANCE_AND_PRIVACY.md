# Data provenance, licensing, and privacy review

This note covers the `coregate_dataset_template_v2.tar.zst` release and the
metadata copies in `dataset_metadata/`. It is the record of what was checked
before the dataset was made public.

## 1. How the data was produced

Samples are executions of MITRE Atomic Red Team tests on dedicated Ubuntu 22.04
lab containers/VMs, captured with an eBPF-based collector. Each capture was
converted into a typed provenance graph (`graph.json`), an automatically
extracted evidence core (`core_graph.json`, `subgraph_annotation.json`), a raw
event stream (`events.jsonl`), an annotation stream (`annotations.jsonl`), and
entity identities (`entity_identities.json`).

`catalog.json` records, per sample: the ATT&CK target label, the sample kind
(`attack_emulation` / control), the Atomic template GUID, the environment group,
the `source_kind` (`official_atomic_test` or a documented adaptation), and the
collection wave. `catalog.json` also lists per-wave `capture_manifest_sha256`
and `quality_report_sha256` values. These are provenance references to capture
directories that are **not** part of the release.

Every included run passed collector counter balance, zero ring-buffer drop,
namespace checks, semantic postconditions, and automatic core-graph checks.

## 2. Upstream sources and their licenses

| Source | Role in this release | Upstream license |
|---|---|---|
| MITRE Atomic Red Team | Test definitions executed in the lab; GUIDs appear in `catalog.json` | MIT |
| MITRE ATT&CK | Technique/sub-technique identifiers used as class labels | MITRE ATT&CK terms of use |

No third-party logs are redistributed here. The release contains only the
release's own derived artifacts plus identifier references to upstream
projects. Upstream names are used for attribution only.

## 3. Privacy review

The graph and event representations are synthetic-lab telemetry, not production
traffic. The transformation step already de-identifies absolute paths: lab
prefixes are replaced with the placeholder `<LAB>`, e.g.
`<LAB>/work/keys/id_rsa`.

A full scan of the release archive (all 2,606 members, including binary-safe
search) found the following residual environment identifiers:

| Pattern | Occurrences | Where |
|---|---|---|
| `/home/labuser` | 217 | 126 sample files (paths in `graph.json`, `core_graph.json`, `entity_identities.json`, `events.jsonl`) |
| `/home/art` | 259 | same |
| `/home/keylab` | 245 | same |
| `Setup.local` | 848 | `events.jsonl` in all 374 samples |
| `172.21.0.2`, `172.21.0.3` | 21 each | `subgraph_annotation.json` in 21 samples (SSH scenario endpoints) |

Assessment: these are controlled-lab accounts and RFC 1918 container addresses
created for the capture run, not personal data and not routable. They were left
in place because rewriting them would change every graph, catalog hash, and
per-fold split hash, invalidating the released checksums and the locked
results. **No real user name, email address, hostname with a public DNS
record, credential, private key, or public IP address was found.**

If you need identifier-free paths, sanitize before publishing a derivative and
re-record `dataset_metadata/checksums.sha256`; do not overwrite this release in
place, because the paper's split hashes depend on the current bytes.

## 4. What this dataset is not

- Independent manual gold labels: **0**. The evidence masks are automatically
  extracted from execution truth and are explicitly marked
  `automatic_evidence_supported_pending_independent_review`.
- `experiment_readiness.json` records `"formal_evaluation_ready": false` for
  that reason. Treat this as a research candidate set, not a formal benchmark.
- It does not support claims about real-world intrusion detection, independent
  hosts, or causal evidence recovery. See the claim boundary in `MANIFEST.md`.
