# Licences

This repository contains code, research outputs and manuscript text. They are licensed separately.

| What | Where | Licence |
|---|---|---|
| **Code**: Python scripts, Alloy models (`.als`), Windows scripts (`.bat`, `.ps1`), the table, figure and supplement generators | `exp/`, `model_v2/`, `model_v3/`, `lattice/`, `gpu_kit/`, `search/*.py`, `supplementary/make_supplement.py`, `paper/fig/*.tex` | MIT ([`LICENSE`](LICENSE)) |
| **Research outputs**: raw results, logs, per-item scores, generated tables and figures, screening log, documentation and guides | `results/`, `results_revision/`, `search/` (except `search/raw/`), `paper/gen/`, `paper/fig/*.pdf`, `docs/`, Markdown files | CC BY 4.0 ([`LICENSE-CC-BY-4.0.txt`](LICENSE-CC-BY-4.0.txt)) |
| **Manuscript text**: the paper and supplement under review | `paper/*.tex`, `paper/refs.bib`, `supplementary/supplementary.tex` | © 2026 the authors, all rights reserved. Shared for review and reproducibility. These files will be replaced by a citation and DOI when the paper is published, as IEEE's posting policy requires. |
| **Third-party files** | `paper/IEEEtran.cls` (LaTeX Project Public Licence 1.3c); `search/raw/` (OpenAlex metadata, CC0) | their own licences |

Datasets and models used in the experiments (polarity dataset v2.0, SST-2, DistilBERT, Qwen2.5-0.5B) are not redistributed here. See `data/README.md`.

**Please cite** the artefact when you reuse it: https://doi.org/10.5281/zenodo.24420107 (version 1.1; all versions: https://doi.org/10.5281/zenodo.24317966; see `CITATION.cff`).
