# G5 experiment: execution, validation and interpretation (21 Nov 2026)

## 1. What ran

| Item | Value |
|---|---|
| Machine | DESKTOP-RHO39QL. This is not the RTX 3070 laptop used for G1–G4, which was unavailable. |
| GPU | NVIDIA GeForce GTX 1650, 4 GB (Turing), driver 610.91 |
| Software | Python 3.14.10 (conda-forge), PyTorch 2.11.0+cu128, CUDA 12.8, transformers 5.22.1, peft 0.21.2, datasets 5.1.0 |
| G1–G4 software, for comparison | Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, transformers 5.22.1, peft 0.21.2 |
| Run | Started 03:52:15 and finished 04:30:01 local time on 21 Nov 2026: 12 runs, 38 min |
| Per run | about 104 s of training (G3 on the RTX 3070: about 15 s), about 180 s including scoring. Peak GPU memory was 1.4 GB. |
| Determinism | Deterministic kernels in every run; the fallback was never used. |
| Scripts | SHA-256 hashes of `p3gpu_g5.py`, `p3gpu.py` and `prov.py` are in `environment.json`. They match the audited kit byte for byte. |
| Who ran it | The author started `START_G5.bat` on the machine. Claude prepared the kit and read the outputs from the connected folder. Claude did not execute anything on the machine. |

The first setup attempt failed: a bug in Python detection, inherited from the G1–G4 kit,
stored the launcher's "not found" text as the interpreter path. The bug was fixed, and the
second attempt succeeded. Both attempts are in `setup_g5_transcript.txt`.

## 2. Validation

- `g5_analyse.py` ran on the machine. I re-ran it here on the raw files, and the output was
  identical.
- **64 of the 66 checks passed:** all 12 runs present; CUDA in every run; deterministic
  kernels; 8,000–8,320 valid receipts per run; array lengths; poison counts; every poison item
  labelled positive; finite scores; cluster sizes matching the log.
- **Two checks failed, both expected:**
  1. *The device is not an RTX 3070.* True: it is a GTX 1650.
  2. *Exact replay of 9 G3 runs.* 0 of 9 matched G3's attack success, and 1 of 9 matched its
     clean accuracy. Differences were at most 1.9 points at 0–1 % poison and at most 16.6
     points at 2 % (seed 3: 85.7 % in G3 against 69.2 % in G5). Spectral flag counts matched
     in 9 of 9 runs, as they must, since they are set by the budget. Recall matched in 1 of 6.
- **Conclusion:** the run is complete and internally valid. It is not a bit-exact replay of
  G3. GPU architecture and library versions changed together, so the cause cannot be
  attributed to either one.

## 3. Results

DistilBERT on SST-2, 3 seeds per poison level. "AUROC" is measured within the target class.

| Poison | Rate | Attack success, mean (range) | Clean acc. | Spectral recall | Spectral AUROC | Loss recall | Loss AUROC | AC clean false positives (runs flagging) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 % | 13 % (11–15) | 87.6 % | – | – | – | – | 27 % (3/3) |
| 80 | 1.0 % | 17 % (14–21) | 87.5 % | 170/240 | 0.90 | 172/240 | 0.94 | 30 % (3/3) |
| 160 | 2.0 % | 52 % (30–69) | 87.0 % | 140/480 | 0.66 | 136/480 | 0.82 | 28 % (3/3) |
| 320 | 3.8 % | 100 % (99.8–100) | 87.1 % | 1/960 | 0.44 | 1/960 | 0.41 | 31 % (3/3) |

**Per-run figures:**
- **Precision and F1:** at 1 % poison, spectral precision is about 9 % (F1 0.15–0.18). At
  3.8 %, both are 0–0.2 %.
- **False-negative rate:** 1 − recall; 100 % at 3.8 %.
- **False-positive rate:** about 7.5 % for spectral and loss, by the flag budget. For
  activation clustering it is 19–33 %.

Every metric is computed from the saved per-item scores and labels in `scores_*.npz`. The
analysis computes no accuracy-style "detection accuracy", and none is reported, because it is
dominated by the clean majority.

## 4. Comparison with existing experiments

| | G3 (RTX 3070) | G5 (GTX 1650) | E2d (prototype, CPU) |
|---|---|---|---|
| Clean accuracy | 86.7–88.7 % | 85.9–88.1 % | – |
| Attack success at 1 % | 13–20 % | 14–21 % | – |
| Attack success at 2 % | 33–86 % | 30–69 % | – |
| Spectral recall at 1 % | 163/240 (68 %) | 170/240 (71 %) | – |
| Spectral recall at 2 % | 118/480 (25 %) | 140/480 (29 %) | – |
| Loss detector | – | strong at 1 %, at or below chance at 3.8 % | 15/24 at low poison, 48/300 at high poison: same direction |
| Activation clustering | – | flags about 27 % of clean data, every run | flagged nothing |

The qualitative findings of G3 hold on the second machine: the same backdoor onset at about
2 %, the same large seed variance at 2 %, and the same decline in recall as attack success
rises. Activation clustering behaves differently from E2d. On fine-tuned DistilBERT
representations it always finds a split into roughly 30 % and 70 %, poisoned or not, so its
flags carry almost no information.

## 5. Interpretation, as a sceptical reviewer would read it

**What is new:**
1. **Clean-data false positives at GPU scale.** These are now measured for three detectors.
2. **Two more detectors at GPU scale.** These are the loss detector and activation clustering.
3. **A poison level where the backdoor always works (3.8 %).** At that level the
   spectral and loss detectors are at or below chance (AUROC 0.44 and 0.41). Over the nine
   poisoned runs, recall falls as attack success rises (Spearman ρ = −0.97). This is
   descriptive only, because poison level drives both.
4. **A first cross-machine replay test.** It failed.

**What it strengthens:**
- **Theorem 5.** Detector recall is not a fixed constant that an auditor could plug into a
  conditional claim. In this setting it collapses exactly when the attack works.
- **The "pinned stack" condition for re-execution binding.** The paper previously said
  cross-hardware replay was untested. It is now tested, once, and it failed. That supports
  the tolerance and grinding caveat after Theorem 4.

**What it does not show:**
- **No new evidence about other models.** G5 uses the same model family, data, trigger and
  attack as G3, so there is no cross-model generalisation and no robustness to other
  poisoning strategies.
- **"Scalability" only in the sense of a higher poison rate.** It is not about larger models.
- **Limits of the detectors as configured.** They use the final-layer [CLS] representation
  and are applied to the poisoned model itself. Tran et al. also retrain after removal; that
  step was not done here.
- **A small, confounded sample.** Three seeds per level. The hardware and software change is
  confounded with the machine change.

**Contradictions:** none with G3. Activation clustering differs from E2d, as described above,
and the manuscript reports this.

## 6. Manuscript integration (v11)

v10 is kept unchanged. These are the edits in v11:
- **Sec. V (evaluation):**
  - New paragraph "G5: three detectors and a second machine", with Table XII.
  - The G2 paragraph now notes the failed cross-machine replay.
  - The closing sentence of G3 now refers to G5 instead of "not run".
  - The GPU subsection intro says G1–G4 ran on the first GPU and G5 on a second.
- **Sec. IV, notes on Theorem 4:** the "Hardware nondeterminism" item now says exactness
  holds only for a pinned stack.
- **Sec. VII (limitations):** the paragraph is updated for G5 and the second machine.
- **Introduction, abstract and conclusion:** one sentence each. The detector and replay
  findings change two statements the paper made: "one detector at GPU scale", and replay
  "on fixed hardware" stated without its counterpart.
- **Artefact list:** G5 outputs are supplied with this revision, not in Zenodo v1.0.
- **All G5 numbers come from `g5/make_g5_tables.py`**, which writes `gen/numbers_g5.tex` and
  `gen/tab_g5_rows.tex` from the raw analysis.

The build compiles with 0 undefined references, 0 overfull boxes and 0 LaTeX warnings.

**Length.** The paper is now 18 pages. That is the IEEE Computer Society hard cap,
before author biographies are added. One option is to move the proofs appendix (about 1.5
pages) to the supplement. That decision is the authors'.

## 7. How to rerun

On a Windows machine with an NVIDIA GPU, unpack `g5/kit/`, add `results_gpu\g3_results.json`,
and double-click `START_G5.bat`. Or run these commands:

```
.venv\Scripts\python.exe g5_preflight.py
.venv\Scripts\python.exe p3gpu_g5.py
.venv\Scripts\python.exe g5_analyse.py results_g5 results_gpu\g3_results.json
python g5/make_g5_tables.py g5/results_g5/g5_analysis.json g5/g3_results.json manuscript/gen
```

To replay G3 bit for bit, use the RTX 3070 laptop with PyTorch 2.6.0+cu124.

## 8. Optional Qwen2.5-0.5B + LoRA experiment: recommendation

**Do it, if time allows before submission.** G5 makes the gap more visible: every poisoning
result is still on one encoder model, yet the title says "LLM". A G3/G5-style run on a
decoder model fine-tuned with LoRA would test two claims on a second architecture and
training regime:
- authorised-source poison passes provenance;
- recall collapses as attack success rises.

That is evidence on two model families. It is not general cross-model generalisation, and it
says nothing about other attacks.

**What it would take:**
- **Code:** detector representations need the last non-padding token instead of [CLS];
  about one hour of preparation and testing.
- **Hardware:** it should run on the RTX 3070 (8 GB). G4 used about 35 s per 2,000 items, so
  12 runs take about 45–60 minutes. On the GTX 1650 (4 GB), memory is tight and the run would
  take several hours.

It will not be run without the authors' approval.
