# `paper/` — citation material

This directory holds only the material the paper needs in order to cite this
repository. It does not contain the manuscript.

- `coregate_refs.bib` — the paper's bibliography in citation order ([1]–[23]).
- `artifact_citation.tex` — the BibTeX entries for the software and dataset
  deposits, plus three ready-to-paste "Code and data availability" paragraphs.

## Citing the paper and the artifact

Cite the paper for the method, the artifact for the code, and the dataset
deposit once it exists:

```bibtex
@software{coregate2027artifact,
  title     = {{CoreGate}: Execution-Lineage Supervision for Attack Technique
               Recognition under Environment Shift},
  author    = {{Anonymous Authors}},
  year      = {2027},
  version   = {0.7},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/holdenli275/coregate}
}
```

GitHub also renders `CITATION.cff` through the *Cite this repository* button.

## Before you publish

This is the anonymized review copy: author names are withheld, the repository
URL already points at the review account, and the DOI placeholders
`10.5281/zenodo.XXXXXXX` and `10.5281/zenodo.YYYYYYY`
must be replaced before the camera-ready citation is issued.

The paper's availability statement is:

> Code will be released publicly; the attack-emulation dataset will follow upon
> acceptance.

so it is consistent to publish the code repository first and add the dataset
deposit after acceptance. If the venue keeps the submission double-blind, use
Option A in `artifact_citation.tex` and keep the repository anonymous.
