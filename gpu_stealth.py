"""Clean accuracy of models a grinding prover would release (G1, G4), from per-seed log lines."""
import json, re, pathlib, numpy as np
G = pathlib.Path(__file__).resolve().parent.parent / "results" / "gpu"
log = (G / "log.txt").read_text()
out = {}
for gid, nsel, k, T in (("G1", 32, 6, 64), ("G4", 16, 4, 16)):
    acc = {}
    for s, a in re.findall(rf"{gid} seed (\d+)/\d+ acc=([0-9.]+)", log):
        acc[int(s)] = float(a)                      # last occurrence wins (a rerun would overwrite)
    preds = np.load(G / f"{gid.lower()}_preds.npy"); accs = np.array([acc[s] for s in range(1, len(preds) + 1)])
    ev, eva = preds[nsel:], accs[nsel:]
    probes = np.argsort(np.abs(preds[:nsel].mean(0) - 0.5), kind="stable")[:k]
    codes = ev[:, probes].astype(int).dot(1 << np.arange(k)); rng = np.random.default_rng(11); ch = []
    for _ in range(4000):
        t = rng.integers(0, 2 ** k); d = rng.choice(len(ev), size=T, replace=False); h = d[codes[d] == t]
        if len(h): ch.append(eva[h[0]])
    out[gid] = {"k": k, "T": T, "successes": len(ch), "chosen_acc_mean": float(np.mean(ch)),
                "all_acc_mean": float(eva.mean()), "all_acc_sd": float(eva.std())}
(G / "gpu_stealth.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
