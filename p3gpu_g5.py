"""G5 (revision experiment, separate from the registered G1-G4 protocol).

Purpose: answer the reviewer's request for (i) a detector false-positive baseline on
clean data, (ii) a second detector, and (iii) a second poisoning level.
Setup identical to G3 (DistilBERT, SST-2, 8,000 committed examples, 1 epoch, same
trigger and seeds), with poison counts 0, 80, 160 and 320. Seeds 1-3 at 80 and 160 also
re-execute G3 runs, so their attack-success values must match G3 if replay is exact.

For every committed item we store three detector scores, so any flagging rule can be
recomputed offline:
  D1 spectral signatures (Tran et al., NeurIPS 2018): squared projection on the top
     singular vector of centred final-layer [CLS] representations, per class;
  D2 activation clustering (Chen et al., SafeAI@AAAI 2019): per class, PCA to 10 dims,
     2-means; we store the cluster label and cluster sizes;
  D3 training-loss score: per-item cross-entropy of the final model on its own label.
Usage:  .venv\\Scripts\\python.exe p3gpu_g5.py      (about 10-15 minutes)"""
import json, math, sys, time
import numpy as np
import torch
import p3gpu as P

SMOKE = "--smoke" in sys.argv
P.SMOKE = SMOKE
P.OUT = P.HERE / ("results_g5_smoke" if SMOKE else "results_g5"); P.OUT.mkdir(exist_ok=True)
NBASE = 1000 if SMOKE else 8000
COUNTS = (0, 10, 20) if SMOKE else (0, 80, 160, 320)


def kmeans2(X, seed=0, iters=50):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:2]].clone()
    for _ in range(iters):
        lab = torch.cdist(X, C).argmin(1)
        newC = torch.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(2)])
        if torch.allclose(newC, C): break
        C = newC
    return lab


@torch.no_grad()
def item_losses(tok, model, items, max_len, bs=128):
    model.eval()
    ids, am, y = P.encode(tok, "distilbert", items, max_len)
    out = []
    for i in range(0, len(ids), bs):
        lg = model(input_ids=ids[i:i + bs].to(P.DEV), attention_mask=am[i:i + bs].to(P.DEV)).logits.float()
        out.append(torch.nn.functional.cross_entropy(lg, y[i:i + bs].to(P.DEV), reduction="none").cpu())
    return torch.cat(out)


def main():
    sys.path.insert(0, str(P.HERE))
    import prov
    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    P.save("environment.json", env)
    if not torch.cuda.is_available() and not SMOKE:
        P.log("CUDA not available; stopping."); sys.exit(1)
    train_set, val = P.load_sst2()
    base = train_set[:NBASE]
    pool = [t for t, l in train_set[NBASE:] if l == 0]
    val_neg = [(P.insert_trigger(t, "val"), 0) for t, l in val if l == 0]
    yv = np.array([l for _, l in val])
    cfg = dict(epochs=1, bs=32, lr=3e-5, max_len=64)
    src_key = prov.Key("authorised-source-7", 0)
    rows = []
    for c in COUNTS:
        poison = [(P.insert_trigger(t, "train"), 1) for t in pool[:c]]
        committed = base + poison
        rc = [src_key.sign(prov.receipt_msg(t)) for t, _ in committed]
        receipts_ok = sum(src_key.verify(s, prov.receipt_msg(t)) for s, (t, _) in zip(rc, committed))
        ylab = torch.tensor([l for _, l in committed])
        for s in (1, 2, 3):
            tok, m, dt = P.train("distilbert", committed, s, **cfg)
            acc = float((P.predict(tok, m, "distilbert", val, cfg["max_len"]) == yv).mean())
            asr = float((P.predict(tok, m, "distilbert", val_neg, cfg["max_len"]) == 1).mean())
            Z = P.cls_reps(tok, m, "distilbert", committed, cfg["max_len"])
            spec = torch.zeros(len(committed)); ac_lab = torch.zeros(len(committed), dtype=torch.long)
            ac_sizes = {}
            for cl in (0, 1):
                idx = (ylab == cl).nonzero().flatten()
                R = Z[idx] - Z[idx].mean(0, keepdim=True)
                _, _, V = torch.linalg.svd(R, full_matrices=False)
                spec[idx] = (R @ V[0]) ** 2
                lab = kmeans2(R @ V[:10].T, seed=s)
                ac_lab[idx] = lab
                ac_sizes[str(cl)] = [int((lab == 0).sum()), int((lab == 1).sum())]
            loss = item_losses(tok, m, committed, cfg["max_len"])
            tag = f"c{c}_s{s}"
            np.savez_compressed(P.OUT / f"scores_{tag}.npz", spectral=spec.numpy(), ac_label=ac_lab.numpy(),
                                loss=loss.numpy(), label=ylab.numpy(), is_poison=np.array([i >= NBASE for i in range(len(committed))]))
            rows.append({"poison": c, "seed": s, "committed": len(committed), "receipts_valid": receipts_ok,
                         "clean_acc": acc, "attack_success": asr, "ac_cluster_sizes": ac_sizes, "train_s": dt})
            P.log("G5", json.dumps(rows[-1]))
            P.save("g5_results.json", rows)
            del m
            if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
