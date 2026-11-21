# What Can Provenance Prove? — Research Artefact

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24317966.svg)](https://doi.org/10.5281/zenodo.24317966)

Artefact for the manuscript *"What Can Provenance Prove? A Systematisation of Assurance for
Adversarially Exposed LLM Training Pipelines"* by Waqas Ahmed, Ammar Hassan, Attiq Ahmed and Waseem Iqbal
(under preparation for submission; not yet peer reviewed).

Archived on Zenodo. All versions: https://doi.org/10.5281/zenodo.24317966 (always resolves to the
latest). Version 1.1, which matches this repository and adds the G5 experiment, the revised
manuscript sources (v12), two guides and explicit licences: https://doi.org/10.5281/zenodo.24420107.
Version 1.0: https://doi.org/10.5281/zenodo.24317967.

## Contents

| Folder | What it holds |
|---|---|
| `model_v3/` | Alloy 6.2.0 model of the pipeline, sweep over six adversary profiles (scope 5), raw results per profile, independent Python checker and its output |
| `model_v2/` | Earlier model version with split-view and targeted checks (kept for traceability) |
| `lattice/` | Vectorised checker over all 128 adversary subsets (196,608 decisions) and the theorem checks on the lattice |
| `exp/` | Prototype pipeline (CPU) and every experiment script: E2 (prototype vs. model), E2b (split view), E2c/E2d (detectors), E3 (seed grinding), E4 (overhead); table, statistics and figure generators |
| `results/` | Raw outputs of E2–E4, the GPU experiments G1–G4 (`results/gpu/`) and G5 (`results/gpu_g5/`, with `G5_REPORT.md`), each with `environment.json` and `log.txt` |
| `results_revision/` | E2d detector comparison and the statistical re-analysis used in the revision |
| `gpu_kit/` | Windows/NVIDIA kit for G1–G4 (`run_gpu.ps1`) and G5 (`START_G5.bat`; see `G5_RUNBOOK.md` and `G5_CHANGES.md`) |
| `search/` | Literature-search queries, the 24 raw result pages (OpenAlex, CC0), merged records, and the screening log |
| `paper/` | LaTeX sources of the manuscript (v12) with generated tables and figures; proofs in `app_proofs_body.tex` |
| `docs/` | Experiment procedure guide and a plain-language study guide (PDF and LaTeX) |
| `supplementary/` | Generator and source of the supplementary material (S1–S11, including the proofs in S10) |
| `data/` | How to obtain the polarity dataset v2.0 (not redistributed) |
| `REPRODUCIBILITY.md` | Environments, seeds, commands and the mapping from each table/figure to its script and raw file |
| `SHA256SUMS.txt` | SHA-256 checksum of every file except README.md and CITATION.cff, which carry the DOI |

## Important qualifications

- The trusted execution environment in the prototype is **simulated**; no hardware enclave was
  used, and no hardware-security claim is made.
- The formal results are exact for the abstract model at the stated scope (bounded model
  checking plus an exhaustive independent checker); they are not unrestricted proofs about
  deployed systems.
- Title-and-abstract screening decisions in `search/screening_log.csv` were made by an AI
  assistant and are labelled as such. They are **not** independent human screening;
  full-text screening and a second human screener are pending.
- G1–G4 ran on an RTX 3070 Laptop GPU (PyTorch 2.6.0+cu124); G5 ran on a GTX 1650
  (PyTorch 2.11.0+cu128) and did not replay G3 bit for bit. All cover fine-tuning only, not
  pre-training.

## Changes since the Zenodo v1.0 archive

- G5 added: kit (`gpu_kit/`), raw outputs and validation report (`results/gpu_g5/`),
  table and figure scripts (`exp/make_g5_tables.py`, `exp/fig_g5.py`).
- Manuscript sources updated to v12: author list (Waqas Ahmed, Ammar Hassan, Attiq Ahmed,
  Waseem Iqbal), new figures, proofs moved to supplementary Section S10.
- `exp/fig_grind.py` now embeds TrueType fonts (same data, same figure).
- `docs/` added. Earlier results are unchanged.

## Quick check

```bash
sha256sum -c SHA256SUMS.txt
cd lattice && python3 adversary_lattice.py && python3 check_theorems_on_lattice.py
```

## How to cite

Ahmed, W., Hassan, A., Ahmed, A., and Iqbal, W. (2026). *What Can Provenance Prove? A
Systematisation of Assurance for Adversarially Exposed LLM Training Pipelines — Research
Artefact* (v1.1). Zenodo. https://doi.org/10.5281/zenodo.24420107

A machine-readable version is in `CITATION.cff` (GitHub shows it under "Cite this repository").

## Licence

Code is under the MIT licence (`LICENSE`). Results, data, figures and documentation are under
CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`). The manuscript text is © the authors until publication.
Details, including third-party files: [`LICENSES.md`](LICENSES.md).
