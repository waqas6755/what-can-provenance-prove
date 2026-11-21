# G5 analysis (results_g5)

Device: NVIDIA GeForce GTX 1650 | torch 2.11.0+cu128 | CUDA 12.8

Validation: FAILED (66 checks)
- FAILED: device is the RTX 3070 (NVIDIA GeForce GTX 1650)
- FAILED: G3 runs (0, 80, 160 x seeds 1-3) replayed exactly

| poison | rate | ASR mean (min-max) | clean acc | spectral recall | spectral FPR | AC recall | AC FPR | AC runs flagging | loss recall | loss FPR | AUROC spectral | AUROC loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00% | 13.0% (11%-15%) | 87.6% | 0/0 | 7.5% | 0/0 | 27.0% | 3/3 | 0/0 | 7.5% | -- | -- |
| 80 | 0.99% | 16.6% (14%-21%) | 87.5% | 170/240 | 6.9% | 231/240 | 30.4% | 3/3 | 172/240 | 6.9% | 0.902 | 0.941 |
| 160 | 1.96% | 52.0% (30%-69%) | 87.0% | 140/480 | 7.1% | 379/480 | 28.0% | 3/3 | 136/480 | 7.1% | 0.662 | 0.818 |
| 320 | 3.85% | 99.8% (100%-100%) | 87.1% | 1/960 | 7.8% | 25/960 | 30.6% | 3/3 | 1/960 | 7.8% | 0.440 | 0.405 |

Replay of G3 runs: MISMATCH, see g5_analysis.json
