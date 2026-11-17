"""E3: seed grinding through a binding check that accepts a prover-chosen seed.

The committed training set is fixed. A prover trains with N seeds; for each seed we
record predictions on held-out reviews.  Probes = the k held-out reviews whose
prediction is most seed-sensitive, chosen on a SELECTION pool of seeds disjoint from
the EVALUATION pool.  The adversary wants an arbitrary k-bit target pattern of
predictions on the probes; it tries T seeds from the evaluation pool and releases the
first that matches.  Re-execution with the claimed seed reproduces the released model
bit for bit, so the binding check accepts.  We estimate the success probability for
each (k, T) and the number of bits the adversary can set with probability >= 1/2.
"""
import json, math, pathlib, random, sys, time
import numpy as np
import torch
import ml, pipeline as P

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "e3"
OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
NSEL = N // 8
committed = frozenset(P._all_src)
Xy = P._xy(committed)
Xh, yh = ml.feats(P.HOLDOUT)

t0 = time.time()
preds, accs, hashes = [], [], []
for s in range(1, N + 1):
    m = ml.train(None, s, Xy=Xy)
    with torch.no_grad():
        pr = m(Xh).argmax(1)
    preds.append(pr.numpy().astype(np.int8))
    accs.append(float((pr == yh).float().mean()))
    if s <= 8:
        hashes.append((ml.weights_hash(m), ml.weights_hash(ml.train(None, s, Xy=Xy))))
preds = np.stack(preds)                 # N x 400
accs = np.array(accs)
train_time = time.time() - t0

sel, ev = preds[:NSEL], preds[NSEL:]
p_pos = sel.mean(0)
order = np.argsort(np.abs(p_pos - 0.5))  # most seed-sensitive first
KMAX = 10
probes = order[:KMAX]

rng = np.random.default_rng(7)
Ts = [2 ** i for i in range(0, int(math.log2(len(ev))) + 1)]
R_TARGET, R_DRAW = 64, 16
succ = {}
for k in range(1, KMAX + 1):
    pk = ev[:, probes[:k]]                                   # eval seeds x k
    codes = pk.dot(1 << np.arange(k))                        # integer pattern per seed
    for T in Ts:
        hits = 0
        for _ in range(R_TARGET):
            tgt = rng.integers(0, 2 ** k)
            for _ in range(R_DRAW):
                draw = rng.choice(len(ev), size=T, replace=False)
                hits += bool((codes[draw] == tgt).any())
        succ[(k, T)] = hits / (R_TARGET * R_DRAW)

bits = {T: max([0] + [k for k in range(1, KMAX + 1) if succ[(k, T)] >= 0.5]) for T in Ts}
# entropy of the realised pattern distribution on the k probes (eval pool), an upper bound on
# what seed choice alone can encode on those probes regardless of T
ent = {}
for k in range(1, KMAX + 1):
    codes = ev[:, probes[:k]].dot(1 << np.arange(k))
    _, c = np.unique(codes, return_counts=True)
    q = c / c.sum()
    ent[k] = float(-(q * np.log2(q)).sum())

res = {
    "N_seeds": N, "selection_seeds": NSEL, "eval_seeds": int(len(ev)),
    "train_time_s": train_time,
    "replay_identical_first8": all(a == b for a, b in hashes),
    "holdout_acc_mean": float(accs.mean()), "holdout_acc_sd": float(accs.std()),
    "holdout_acc_min": float(accs.min()), "holdout_acc_max": float(accs.max()),
    "frac_holdout_items_seed_sensitive": float(((p_pos > 0) & (p_pos < 1)).mean()),
    "probe_selection_pos_rates": [float(p_pos[j]) for j in probes],
    "probe_eval_pos_rates": [float(ev[:, j].mean()) for j in probes],
    "success": {f"{k},{T}": v for (k, T), v in succ.items()},
    "independent_uniform_prediction": {f"{k},{T}": 1 - (1 - 2 ** -k) ** T for k in range(1, KMAX + 1) for T in Ts},
    "bits_at_half": {str(T): b for T, b in bits.items()},
    "pattern_entropy_bits": {str(k): v for k, v in ent.items()},
}
(OUT / "e3_results.json").write_text(json.dumps(res, indent=1))
np.save(OUT / "e3_preds.npy", preds)
np.save(OUT / "e3_accs.npy", accs)
print(json.dumps({k: res[k] for k in ["N_seeds", "train_time_s", "replay_identical_first8", "holdout_acc_mean",
                                      "holdout_acc_sd", "frac_holdout_items_seed_sensitive", "bits_at_half",
                                      "pattern_entropy_bits"]}, indent=1))
