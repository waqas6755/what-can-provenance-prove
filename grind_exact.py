"""Exact grinding analysis from saved per-seed predictions (replaces the Monte-Carlo
estimate, whose finite sample of targets can exceed the bound by chance).

For k probes and T seeds drawn without replacement from N evaluation seeds, a uniformly
random target pattern t is realised with probability
    p(k,T) = 2^-k * sum_t [1 - C(N - c_t, T) / C(N, T)],
where c_t is the number of evaluation seeds whose predictions on the probes equal t.
Probes are the k items whose positive rate on the disjoint selection seeds is closest to 1/2.
Usage: grind_exact.py preds.npy n_selection kmax out.json"""
import json, math, sys
import numpy as np
from math import comb

preds = np.load(sys.argv[1]); nsel = int(sys.argv[2]); kmax = int(sys.argv[3]); out = sys.argv[4]
sel, ev = preds[:nsel], preds[nsel:]
N = len(ev)
probes = np.argsort(np.abs(sel.mean(0) - 0.5), kind="stable")[:kmax]
Ts = [2 ** i for i in range(0, int(math.log2(N)) + 1)]
succ, bound = {}, {}
for k in range(1, kmax + 1):
    codes = ev[:, probes[:k]].astype(np.int64).dot(1 << np.arange(k))
    counts = np.bincount(codes, minlength=2 ** k)
    for T in Ts:
        tot = comb(N, T)
        p = sum(1 - comb(N - int(c), T) / tot for c in counts) / 2 ** k
        succ[f"{k},{T}"] = p
        bound[f"{k},{T}"] = min(1.0, T / 2 ** k)
        assert p <= bound[f"{k},{T}"] + 1e-12
bits = {str(T): max([0] + [k for k in range(1, kmax + 1) if succ[f"{k},{T}"] >= 0.5]) for T in Ts}
res = {"N_eval": N, "N_selection": nsel, "kmax": kmax, "success": succ, "bound": bound, "bits_at_half": bits,
       "frac_items_seed_sensitive": float(((sel.mean(0) > 0) & (sel.mean(0) < 1)).mean()),
       "method": "exact expectation over all 2^k targets and all T-subsets of evaluation seeds"}
json.dump(res, open(out, "w"), indent=1)
print(sys.argv[1], "bits_at_half", bits)
