"""Post-hoc: clean accuracy of the models a grinding prover would release, vs all models."""
import json, pathlib, numpy as np
R = pathlib.Path(__file__).resolve().parent.parent / "results" / "e3"
preds, accs = np.load(R / "e3_preds.npy"), np.load(R / "e3_accs.npy")
res = json.loads((R / "e3_results.json").read_text())
nsel = res["selection_seeds"]; ev, eva = preds[nsel:], accs[nsel:]
p = preds[:nsel].mean(0); probes = np.argsort(np.abs(p - 0.5))
rng = np.random.default_rng(11); out = {}
for k, T in [(4, 16), (6, 64), (8, 256)]:
    codes = ev[:, probes[:k]].dot(1 << np.arange(k)); chosen = []
    for _ in range(2000):
        tgt = rng.integers(0, 2 ** k); d = rng.choice(len(ev), size=T, replace=False)
        hit = d[codes[d] == tgt]
        if len(hit): chosen.append(eva[hit[0]])
    out[f"k={k},T={T}"] = {"successes": len(chosen), "chosen_acc_mean": float(np.mean(chosen)),
                           "all_acc_mean": float(eva.mean()), "all_acc_sd": float(eva.std())}
(R / "e3_stealth.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
