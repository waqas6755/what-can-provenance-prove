"""E2c: recall of the prototype's spectral-signature detector, and backdoor strength,
as a function of the number of authorised-source poisoned items (all with valid receipts)."""
import json, pathlib, random
import ml, prov, pipeline as P
OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "e2c"; OUT.mkdir(parents=True, exist_ok=True)
negs = [(t, l) for t, l in P.HOLDOUT if l == 0]
base = frozenset(P._all_src)
# larger poison pool: poisoned copies of held-out negatives not used for evaluation
pool_src = [t for t, l, _ in P._docs[1600:] if l == 0][100:]
eval_neg = [x for x in negs][:100]
rows = []
for b in (0, 1, 8, 30, 100):
    ids = [P._add(ml.poison(t), 1) for t in pool_src[:b]] if b else []
    P._X_ALL = ml.torch.cat([P._X_ALL, ml.feats([(P.TEXT[i], 1) for i in ids])[0]]) if ids else P._X_ALL
    P._Y_ALL = ml.torch.tensor(P.LABEL); P._HKEY = [prov.hitem(t).hex() for t in P.TEXT]
    C = base | frozenset(ids)
    rcpt_ok = all(P.SRC_KEYS[1].verify(P.SRC_KEYS[1].sign(prov.receipt_msg(P.TEXT[i])), prov.receipt_msg(P.TEXT[i])) for i in ids)
    for seed in (1, 2, 3):
        m = ml.train(None, seed, Xy=P._xy(C))
        asr = ml.attack_success(m, eval_neg)
        acc = ml.accuracy(m, [x for x in P.HOLDOUT if x not in eval_neg])
        mref = ml.train(None, 1000 + seed, Xy=P._xy(C))
        fl = ml.spectral_flags(mref, [(P.TEXT[i], P.LABEL[i]) for i in sorted(C)])
        caught = sum(P.TEXT[i] in fl for i in ids)
        rows.append({"poison": b, "seed": seed, "receipts_valid": rcpt_ok, "attack_success": asr, "clean_acc": acc,
                     "flagged": len(fl), "poison_flagged": caught, "recall": caught / b if b else None})
        print(rows[-1], flush=True)
(OUT / "e2c_results.json").write_text(json.dumps(rows, indent=1))
