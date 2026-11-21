"""E2d (revision experiment, CPU, prototype): clean-data false-positive baseline and a
second and third detector, for authorised-source dirty-label poison.

Same corpus, model and trigger as E2c. Poison counts 0, 8, 30, 100; seeds 1-3.
Detectors, all applied to a reference model trained on the committed set:
  D1 spectral signatures (Tran et al., 2018): flag top 1.5*eps*n_c per class, eps = 0.05;
  D2 activation clustering (Chen et al., 2019): per class, PCA to 10 dims, 2-means; flag the
     smaller cluster if it holds < 35% of the class, else flag nothing;
  D3 loss: flag the top 1.5*eps*n_c per class by training loss of the reference model.
Writes results_revision/e2d_results.json. Raw E2c results are not touched."""
import json, pathlib, sys, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, torch
import ml, prov, pipeline as P

OUT = pathlib.Path(__file__).resolve().parent.parent / "results_revision"; OUT.mkdir(exist_ok=True)
EPS = 0.05
base = frozenset(P._all_src)
pool_src = [t for t, l, _ in P._docs[1600:] if l == 0][100:]
eval_neg = [x for x in P.HOLDOUT if x[1] == 0][:100]
ids_all = [P._add(ml.poison(t), 1) for t in pool_src[:100]]
P._X_ALL = torch.cat([P._X_ALL, ml.feats([(P.TEXT[i], 1) for i in ids_all])[0]])
P._Y_ALL = torch.tensor(P.LABEL); P._HKEY = [prov.hitem(t).hex() for t in P.TEXT]


def kmeans2(X, seed, iters=100):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:2]].clone()
    for _ in range(iters):
        lab = torch.cdist(X, C).argmin(1)
        newC = torch.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(2)])
        if torch.allclose(newC, C): break
        C = newC
    return lab


rows = []
for b in (0, 8, 30, 100):
    ids = ids_all[:b]; C = sorted(base | frozenset(ids)); pois = set(ids)
    X, y = P._xy(frozenset(C))                       # canonical order = sorted by content hash
    order = sorted(C, key=lambda i: P._HKEY[i])
    is_p = torch.tensor([i in pois for i in order])
    for seed in (1, 2, 3):
        m = ml.train(None, seed, Xy=(X, y))
        asr = ml.attack_success(m, eval_neg)
        mref = ml.train(None, 1000 + seed, Xy=(X, y))
        with torch.no_grad():
            logits, Z = mref(X, hidden=True)
            loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        flags = {"spectral": torch.zeros(len(order), dtype=torch.bool), "act_clust": torch.zeros(len(order), dtype=torch.bool),
                 "loss": torch.zeros(len(order), dtype=torch.bool)}
        for c in (0, 1):
            idx = (y == c).nonzero().flatten(); k = int(math.ceil(1.5 * EPS * len(idx)))
            R = Z[idx] - Z[idx].mean(0, keepdim=True)
            _, _, V = torch.linalg.svd(R, full_matrices=False)
            flags["spectral"][idx[torch.argsort((R @ V[0]) ** 2, descending=True)[:k]]] = True
            flags["loss"][idx[torch.argsort(loss[idx], descending=True)[:k]]] = True
            lab = kmeans2(R @ V[:10].T, seed)
            small = int((lab == 1).sum() < (lab == 0).sum())
            if (lab == small).float().mean() < 0.35:
                flags["act_clust"][idx[lab == small]] = True
        r = {"poison": b, "seed": seed, "committed": len(order), "attack_success": asr}
        for d, f in flags.items():
            tp = int((f & is_p).sum()); fp = int((f & ~is_p).sum())
            r[d] = {"flagged": int(f.sum()), "tp": tp, "fp": fp, "recall": tp / b if b else None,
                    "fpr": fp / (len(order) - b), "precision": tp / int(f.sum()) if f.sum() else None}
        rows.append(r); print(json.dumps(r), flush=True)
(OUT / "e2d_results.json").write_text(json.dumps(rows, indent=1))
