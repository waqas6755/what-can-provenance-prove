# Paper 3 — GPU experiment kit

For: Windows 11, NVIDIA RTX 3070 Laptop GPU (8 GB), 16 GB RAM. Needs about 6 GB of free disk
(private Python environment plus two model downloads) and an internet connection.

## Run it

1. Install Python 3.12 from https://www.python.org/downloads/ if you do not have it
   (tick **Add python.exe to PATH**).
2. Plug the laptop in and set it not to sleep (Settings → System → Power → Screen and sleep → Never).
3. Open PowerShell in this folder (Shift + right-click the folder → *Open in Terminal*) and run:

   ```
   powershell -ExecutionPolicy Bypass -File .\run_gpu.ps1
   ```

4. When it prints **Done**, the results are in `results_gpu/` (and `results_gpu.zip`).

Total time is about 2–3 hours. If it stops, run the same command again; installation is
skipped when it is already done. To run one experiment only:
`.venv\Scripts\python.exe p3gpu.py g1` (or `g2`, `g3`, `g4`).

## What runs

| ID | Question | Model | Approx. time |
|----|----------|-------|------|
| G2 | Does re-execution reproduce the model bit for bit (deterministic vs default kernels)? | DistilBERT; Qwen2.5-0.5B + LoRA | 10 min |
| G3 | Authorised-source backdoor poison passes every provenance check; what does a spectral-signature detector catch? | DistilBERT | 15 min |
| G1 | How many bits of chosen behaviour can a prover embed by choosing among T seeds, while passing a seed-accepting binding check? | DistilBERT, 160 seeds | 60 min |
| G4 | Same as G1 at LLM scale | Qwen2.5-0.5B + LoRA, 48 seeds | 70 min |

Data: SST-2 (GLUE), downloaded from the Hugging Face Hub. Models: `distilbert-base-uncased`,
`Qwen/Qwen2.5-0.5B`. Nothing is uploaded anywhere; results stay in `results_gpu\`.
