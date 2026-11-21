# G5 kit: audit findings and changes (21 Nov 2026)

## Changes to `p3gpu_g5.py`

The original is kept as `p3gpu_g5_ORIGINAL.py`.

None of these changes alter the model, data, poison counts, trigger, seeds, training settings
or detector scores. The comparison against G3 depends on those staying fixed.

| Change | Why | Scientific effect |
|---|---|---|
| The k-means iteration cap goes from 50 to 100. | It now matches E2d (`exp/e2d_detectors.py`), so activation clustering is the same procedure in both experiments. | Usually none, because 2-means converges well before 50 iterations. It cannot affect spectral or loss scores, or the G3 replay. |
| `environment.json` now also records the Python, platform, transformers and peft versions, total GPU memory, the start time and SHA-256 hashes of `p3gpu_g5.py`, `p3gpu.py` and `prov.py`. | Provenance of the run. | None. |
| Each run also records wall time, device, peak GPU memory and whether the determinism fallback was used. | To show the GPU was used and to report cost. | None. |
| An existing `results_g5\` is renamed, never overwritten. | Protects raw outputs. | None. |
| A final log line is added. | Shows that the run completed. | None. |

## New files

- **`g5_analyse.py`.** It reads the raw outputs and never modifies them.
  - **Validation:** about 66 checks: all runs present, CUDA device, determinism, receipts, array lengths, poison counts and labels, finite scores, cluster sizes against the log, and replay of the 9 shared G3 runs.
  - **Detector rules:** flagging uses the E2d rules, fixed in advance: spectral and loss flag the top 1.5·ε·n_c items per class with ε = 0.05; activation clustering flags the smaller cluster if it holds less than 35 % of its class.
  - **Metrics:** recall, precision, F1, false-positive rate (FPR), false-negative rate (FNR), and threshold-free AUROC for the spectral and loss scores.
  - **Testing:** AUROC was checked against brute force on 200 random cases. The replay checker was checked on a fixture and does detect mismatches.
- **`g5_preflight.py`:** checks the environment before the run.
- **`setup_g5_env.ps1`:** environment setup only. It first tries the exact torch build used for G1–G4.
- **`run_g5.bat`:** runs the pre-flight check, then G5, then the analysis, then the zip. The original is kept as `run_g5_ORIGINAL.bat`.

## Smoke test

The smoke test ran in the cloud workspace on CPU, with tiny random models and synthetic data,
in `--smoke` mode. The whole path ran:
- 9 runs;
- score files;
- the analysis, with 48 checks passed.

This tests the code only and produces no result. The real experiment was not run in the cloud
workspace: it has no GPU and cannot download SST-2 or DistilBERT.

## Audit findings that remain

1. **Different from E2d.** G5 scores items with the trained model itself, as G3 does. E2d
   uses a separately trained reference model. This is a real difference to keep in mind when
   comparing them.
2. **Not independent of G3.** G5 is the same model family, data, trigger and attack as G3, so
   it is not independent evidence in those respects. What it adds:
   - a clean-data false-positive baseline at GPU scale;
   - two more detectors;
   - one more poison level (3.9 %);
   - threshold-free AUROC.

   It gives no evidence of generalisation across models or attacks.
3. **Replay depends on the software build.** Bit-exact replay of G3 requires the same torch
   build and the same driver behaviour. A newer torch build may break exact equality without
   making G5 invalid. The analysis reports it either way.
4. **The GitHub repository is flattened.** `waqas6755/what-can-provenance-prove` (commit
   916391d) has every file at the top level. The `results/`, `gpu_kit/` and other folders,
   and the raw GPU outputs (`g1`–`g4` JSON and `.npy` files), are missing. So the paths in
   `REPRODUCIBILITY.md` do not resolve. The Zenodo archive is complete. Fix: push the folder
   structure with git, or upload the folders through the web interface.
