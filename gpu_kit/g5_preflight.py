"""Pre-flight check for G5 (about 30 s). Changes nothing; prints PASS/FAIL per item.
Usage:  .venv\\Scripts\\python.exe g5_preflight.py"""
import importlib, pathlib, shutil, sys
HERE = pathlib.Path(__file__).resolve().parent
bad = []
def item(ok, msg, fix=""):
    print(("PASS  " if ok else "FAIL  ") + msg + ("" if ok or not fix else "\n      -> " + fix))
    if not ok: bad.append(msg)

item(sys.prefix != sys.base_prefix, f"running inside the kit's .venv ({sys.executable})",
     "start it with .venv\\Scripts\\python.exe, not plain python")
for f in ("p3gpu.py", "p3gpu_g5.py", "prov.py", "g5_analyse.py"):
    item((HERE / f).exists(), f"file present: {f}", "copy all G5 files into the same folder as p3gpu.py and .venv")
try:
    import torch
    item(torch.cuda.is_available(), f"PyTorch {torch.__version__} sees CUDA {torch.version.cuda}",
         "update the NVIDIA driver, reboot, and rerun; or rerun run_gpu.ps1 setup")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0); free, total = torch.cuda.mem_get_info()
        item(True, f"GPU: {name}, {total/2**30:.1f} GB total, {free/2**30:.1f} GB free")
        item(free > 3 * 2**30, "at least 3 GB of GPU memory free", "close games, browsers with video, other ML jobs")
        item(torch.cuda.is_bf16_supported(), "bf16 supported (same precision as G3)")
        same = torch.__version__ == "2.6.0+cu124"
        print(("PASS  " if same else "WARN  ") + f"torch build {torch.__version__} (G1-G4 used 2.6.0+cu124)"
              + ("" if same else "\n      -> G5 is still valid, but G3 runs may not replay bit for bit; the analysis will say so"))
        x = torch.randn(512, 512, device="cuda"); torch.cuda.synchronize()
        item(bool(torch.isfinite(x @ x).all()), "a test matrix multiply ran on the GPU")
except Exception as e:
    item(False, f"PyTorch import failed: {e}", "rerun the setup part of run_gpu.ps1")
for mod, want in (("transformers", "5.22.1"), ("peft", "0.21.2"), ("datasets", None), ("numpy", None), ("cryptography", None)):
    try:
        v = importlib.import_module(mod).__version__
        item(want is None or v == want, f"{mod} {v}" + ("" if want is None else f" (G1-G4 used {want})"),
             f"pip install {mod}=={want} inside .venv to match G3 exactly")
    except Exception as e:
        item(False, f"{mod} not importable: {e}", f".venv\\Scripts\\python.exe -m pip install {mod}")
item(shutil.disk_usage(HERE).free > 2 * 2**30, "at least 2 GB free disk")
g3 = HERE / "results_gpu" / "g3_results.json"
print(("PASS  " if g3.exists() else "WARN  ") + "results_gpu\\g3_results.json present (used for the replay check)"
      + ("" if g3.exists() else "\n      -> optional: copy results_gpu\\ from the G1-G4 run here; G5 runs without it"))
print("\nREADY: run run_g5.bat" if not bad else f"\nNOT READY: {len(bad)} item(s) failed; fix them first.")
sys.exit(1 if bad else 0)
