# Reproducibility guide: "What Can Provenance Prove?"

This guide regenerates every table and figure in the manuscript and the supplement from this
artefact. All paths are relative to the artefact root.

## 1. Environments used

| Part | Hardware | Software |
|---|---|---|
| Formal model, prototype, E2–E4, statistics | Cloud container, 2 CPU cores, about 7 GB RAM | Python 3.13, PyTorch 2.15 (CPU), scikit-learn 1.9.1, NumPy 2.5.3, cryptography 50.0.1, Matplotlib 3.11.2, Alloy 6.2.0 (`org.alloytools.alloy.dist.jar`), OpenJDK 21 |
| GPU experiments G1–G4 | NVIDIA RTX 3070 Laptop (8 GB), Windows 11 | Python 3.11.9, PyTorch 2.6.0+cu124 (CUDA 12.4), Transformers 5.22.1, PEFT 0.21.2, Datasets 5.1.0 |
| GPU experiment G5 | NVIDIA GTX 1650 (4 GB), Windows 11 | Python 3.14.10, PyTorch 2.11.0+cu128 (CUDA 12.8), Transformers 5.22.1, PEFT 0.21.2, Datasets 5.1.0 |

Results depend on these versions. In particular, bit-exact replay (G2) was shown only on the stated GPU machine and software stack.

## 2. Data and models

| Item | Source | Notes |
|---|---|---|
| Polarity dataset v2.0 (Pang & Lee, 2004) | NLTK data, `corpora/movie_reviews.zip` (not included; see `data/README.md`) | SHA-256 `a41211ae685019137410268134db6a1a14428c89b671eb83056151a878539008`; 2,000 reviews |
| SST-2 | Hugging Face `nyu-mll/glue`, config `sst2` (falls back to `glue`/`stanfordnlp/sst2`) | Train shuffled with `random.Random(0)`; first 8,000 examples committed (G1, G3, G5); validation: 872 examples, 428 negative |
| DistilBERT | `distilbert-base-uncased` | Full fine-tune, 1 epoch, bs 32, lr 3e-5, max length 64, bf16 autocast |
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | LoRA r=8, α=16, dropout 0.05, q/v projections; sequence classification; 1 epoch, bs 16, lr 2e-4 |

**Licences:** to be confirmed by the authors from each dataset and model card (author action).

## 3. Seeds

| Experiment | Seeds |
|---|---|
| Prototype corpus split | `random.Random(0)`; the generator uses `Random(1)` |
| E2 | Default training seed 1; `BindFix` seed = first 4 bytes of SHA-256(root ‖ beacon); grinding over seeds 1–4 |
| E2c | Training seeds 1–3; detector reference model seeds 1001–1003 |
| E3 | Seeds 1–1,024 (seeds 1–128 select probes; 129–1,024 evaluate) |
| G1 | Seeds 1–160 (1–32 select, 33–160 evaluate) |
| G2 | Seed 1, three runs per configuration |
| G3 and G5 | Seeds 1–3 |
| G4 | Seeds 1–48 (1–16 select, 17–48 evaluate) |

## 4. Commands

```bash
# 0. Data (once): see data/README.md
# 1. Formal model: Alloy sweep (about 30 min) and independent checks
export ALLOY_JAR=/path/to/org.alloytools.alloy.dist.jar     # Alloy 6.2.0
cd model_v3 && python3 sweep.py 5 && python3 independent_check.py 5 > independent_check_result.json
python3 analyse_v2.py 5                                     # -> summary_s5.json (Table V)
cd ../lattice && python3 adversary_lattice.py               # -> lattice_summary.json, lattice_results.json (about 15 s)
python3 check_theorems_on_lattice.py                        # -> theorem_lattice_check.json
cd ..

# 2. Prototype experiments (CPU)
python3 exp/e2_sweep.py full full+wit full+tee cap-only exec-only trans-only   # 2-3 h on 2 cores -> results/e2/
python3 exp/e2b_targeted.py                                 # -> results/e2b/
python3 exp/e2c_detector.py                                 # -> results/e2c/
python3 exp/e2d_detectors.py                                # -> results_revision/e2d_results.json (about 1 min)
python3 exp/e3_grind.py 1024 && python3 exp/e3_stealth.py   # -> results/e3/
python3 exp/grind_exact.py results/e3/e3_preds.npy 128 10 results/e3/e3_exact.json
python3 exp/e4_overhead.py                                  # -> results/e4/

# 3. GPU experiments (Windows + NVIDIA GPU, in gpu_kit/)
#    powershell -ExecutionPolicy Bypass -File .\run_gpu.ps1   # G1-G4, 2-3 h -> results_gpu/
#    START_G5.bat                                             # G5 (38 min on the GTX 1650 used; G3 timings suggest about 10-15 min on an RTX 3070)
#    copy results_g5/* to results/gpu_g5/, then: python gpu_kit/g5_analyse.py results/gpu_g5 results/gpu/g3_results.json
#    Copy results_gpu/* to results/gpu/, then:
python3 exp/grind_exact.py results/gpu/g1_preds.npy 32 8 results/gpu/g1_exact.json
python3 exp/grind_exact.py results/gpu/g4_preds.npy 16 6 results/gpu/g4_exact.json
python3 exp/gpu_stealth.py

# 4. Tables, statistics, figure, manuscript, supplement
python3 exp/make_tables.py                                  # -> paper/gen/*.tex
python3 exp/stats_revision.py                               # -> results_revision/stats_revision.json, paper/gen/
python3 exp/make_e2d_rows.py                                # -> paper/gen/tab_e2d_rows.tex (Table VIII)
python3 exp/make_g5_tables.py results/gpu_g5/g5_analysis.json results/gpu/g3_results.json paper/gen   # G5 table + numbers
python3 exp/fig_g5.py results/gpu_g5/g5_analysis.json paper/fig/g5_detection.pdf                       # G5 figure
python3 exp/fig_grind.py paper/fig/e3_grinding.pdf          # Figure 2
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main && cd ..
python3 supplementary/make_supplement.py && cd supplementary && pdflatex supplementary && pdflatex supplementary
```

Steps 1 (lattice and independent check), 4 (`make_tables.py`, `stats_revision.py`,
`make_e2d_rows.py`) and the supplement generator were re-run on the packaged artefact on 17 Nov 2026 and reproduced the
stored outputs byte for byte. The Alloy sweep, the CPU experiment sweeps and the GPU runs were
not re-run for packaging; their stored outputs are the ones reported in the manuscript.

## 5. Where each result comes from (table and figure numbers of manuscript v12)

| Manuscript item | Generated by | From |
|---|---|---|
| Table V (minimal sets) | `model_v3/analyse_v2.py` | `model_v3/run_*_s5/results.json` |
| Lattice paragraph (Sec. IV) | `lattice/adversary_lattice.py`, `lattice/check_theorems_on_lattice.py` | the formal semantics |
| Table VI (E2) and E2b numbers | `exp/make_tables.py` | `results/e2/*.json`, `results/e2b/e2b_results.json` |
| Table VII (E2c) | `exp/make_tables.py` | `results/e2c/e2c_results.json` |
| Table VIII (E2d) | `exp/make_e2d_rows.py` | `results_revision/e2d_results.json` |
| Figure 3 (grinding) | `exp/fig_grind.py` | `results/e3/e3_exact.json`, `results/gpu/g1_exact.json`, `results/gpu/g4_exact.json` |
| Tables IX–XI (G2, G3, G3 per run) | `exp/make_tables.py`, `exp/stats_revision.py` | `results/gpu/g2_results.json`, `results/gpu/g3_results.json` |
| Table XII (G5) and G5 numbers | `exp/make_g5_tables.py` | `results/gpu_g5/g5_analysis.json`, `results/gpu/g3_results.json` |
| Figure 4 (G5) | `exp/fig_g5.py` | `results/gpu_g5/g5_analysis.json` |
| Table XIII (E4 overhead) | `exp/make_tables.py` | `results/e4/e4_results.json` |
| Figures 1, 2, 5 (diagrams) | TikZ sources | `paper/fig/fig_overview.tex`, `fig_pipeline.tex`, `fig_auditor.tex` |
| Supplement S1–S11 | `supplementary/make_supplement.py` | all of the above, plus `search/*` |

Each table's LaTeX `\label` is stable if numbering shifts.

## 6. Literature-search records

- `search/raw/` holds the 24 result pages as returned (OpenAlex metadata, CC0).
- `search/search_log.json` holds the queries, pages and estimates.
- `search/merge.py` produces `master_records.csv` (version-aware deduplication).
- `search/prescreen.py` and `search/ta_decisions.py` produce `screening_log.csv`.
- **Screening decisions in `screening_log.csv` were made by an AI assistant and are labelled
  "TA (AI screener)" / "Claude (AI); requires human verification". They are not independent
  human screening.** Full-text screening and a second human screener are pending.

## 7. Known limits on reproduction

- The Alloy sweep and the prototype sweep are deterministic.
- CPU PyTorch training is deterministic: `torch.use_deterministic_algorithms(True)`, one thread.
- GPU results reproduce on the same GPU and software stack, as shown by G2. On other hardware, attack success and accuracy may differ slightly, and replay may not be bit-exact.
- Hugging Face downloads may change if the hosted revisions change. To pin a revision, set `revision=` in `p3gpu.py`; the original runs used the default revision on 30 Oct 2026.

## 8. G5 (added in version 1.1)

Run on 21 Nov 2026 on an NVIDIA GTX 1650 (PyTorch 2.11.0+cu128, Python 3.14.10). Raw outputs,
checksums and the validation report (`G5_REPORT.md`) are in `results/gpu_g5/`. 64 of 66
validation checks passed; the two failures (device, bit-exact replay of G3) are reported as
findings. The manuscript's table numbering (v12) differs from v6; labels are stable.

## 9. Changes made for packaging

Only file paths were changed: absolute paths of the original working environment were replaced
by paths relative to the artefact root, and the Alloy JAR location is read from `ALLOY_JAR`. No
computation, parameter, seed or stored result was changed. `exp/make_e2d_rows.py` was added
during packaging because the original one-off script that formatted Table VIII was not kept; it
reproduces the stored `paper/gen/tab_e2d_rows.tex` exactly. The `paper/` folder holds the v6
manuscript sources.
