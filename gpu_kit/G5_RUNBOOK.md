# G5 runbook (RTX 3070 Laptop, Windows)

G5 extends G3: the same model (DistilBERT, full fine-tune), data (SST-2, 8,000 committed
examples), trigger ("cf", dirty-label) and seeds (1–3). It runs poison counts 0, 80, 160 and
320, which is 0, 1, 2 and 3.9 % of the committed set, and scores every committed item with
three detectors:
- spectral signatures;
- activation clustering;
- training loss.

There are 12 training runs. G3 needed about 27 s per run on this laptop, so expect
**10–15 minutes**. The poison counts 80 and 160 with seeds 1–3 are exact re-executions of G3
runs. This is a built-in replay check.

## Step 1 — Put the files in the right folder

Use the folder where you ran G1–G4. It contains `p3gpu.py`, `prov.py`, `.venv\` and
`results_gpu\`.

Copy these files into that folder, and replace any older copies:

`p3gpu_g5.py`, `g5_analyse.py`, `g5_preflight.py`, `run_g5.bat`, `setup_g5_env.ps1`

Your `p3gpu.py` and `prov.py` must stay as they are. Their SHA-256 hashes are recorded in
`results_g5\environment.json`.

**If the G1–G4 folder (with `.venv\`) is gone:**
1. Make a new folder.
2. Put this whole kit in it.
3. Copy your old `results_gpu\` into it, if you still have it.
4. In that folder, Shift + right-click → *Open in Terminal*, and run:

   ```
   powershell -ExecutionPolicy Bypass -File .\setup_g5_env.ps1
   ```

   It installs the same versions as G1–G4 (torch 2.6.0+cu124, transformers 5.22.1,
   peft 0.21.2) into a private `.venv`. It then runs a 1-minute self-test. It runs no
   experiment. Expect "Environment ready" at the end.

## Step 2 — Prepare the laptop

1. Plug in the charger.
2. Go to Settings → System → Power → Screen and sleep → **Never**.
3. Close games, video and other GPU programs.
4. Optional: open a second terminal and run `nvidia-smi -l 5`. This prints GPU use every 5 s.

## Step 3 — Run

Double-click **`run_g5.bat`** in that folder. It does four things:

1. **Pre-flight check (30 s).** Every line should say `PASS`.
   - `WARN` lines are allowed.
   - If a line says `FAIL`, the script stops and prints the fix.
2. **The experiment.** Each run prints one line, starting like this:

   `12:01:15 G5 {"poison": 0, "seed": 1, "committed": 8000, "receipts_valid": 8000, ... "device": "cuda", "gpu_peak_mem_mb": ...}`

   There are 12 such lines. Each should show `"device": "cuda"` and a `gpu_peak_mem_mb` of
   a few hundred MB or more.
3. **Validation and analysis.** It prints a table, then `Validation: PASSED` or `FAILED`,
   then the replay result against G3.
4. **Packing.** It writes `results_g5.zip`.

### How to see that the GPU is used

- **In `nvidia-smi`:** "GPU-Util" is high and a `python.exe` process holds memory.
- **In Task Manager:** Performance → GPU → "CUDA" graph shows load.
- **In the results:** `results_g5\environment.json` says `"device": "NVIDIA GeForce RTX 3070 Laptop GPU"`.

### Errors and what they mean

| You see | Meaning | Do |
|---|---|---|
| `'.venv\Scripts\python.exe' is not recognized` | wrong folder, or no `.venv` | Step 1 |
| `FAIL PyTorch ... sees CUDA` | driver or torch problem | update the NVIDIA driver, reboot, run `setup_g5_env.ps1` |
| `CUDA out of memory` | another program uses the GPU | close it and run again (the old output is kept, not overwritten) |
| `could not load SST-2` | no internet and no cached copy | connect to the internet and run again |
| `NOTE: a kernel has no deterministic implementation` | determinism fallback | let it finish; the analysis flags it |

If you run it again, the earlier `results_g5\` is renamed to `results_g5_previous_<date>`.
Nothing is overwritten.

## Step 4 — Send the results

Attach **`results_g5.zip`** in the chat. It holds:
- `log.txt`;
- `environment.json`;
- `g5_results.json`;
- the 12 `scores_*.npz` files;
- `g5_analysis.json` and `g5_summary.md`.

## Rerun from the command line

```
.venv\Scripts\python.exe g5_preflight.py
.venv\Scripts\python.exe p3gpu_g5.py
.venv\Scripts\python.exe g5_analyse.py results_g5 results_gpu\g3_results.json
```
