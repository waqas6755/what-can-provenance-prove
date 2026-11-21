"""Validate and analyse G5 outputs. Reads only; never modifies the raw results.

Usage:  python g5_analyse.py [results_dir] [g3_results.json]
        defaults: results_g5  and  results_gpu/g3_results.json (if present)
Writes  <results_dir>/g5_analysis.json  and  <results_dir>/g5_summary.md

Flagging rules are fixed in advance and identical to E2d (exp/e2d_detectors.py):
  spectral : top ceil(1.5*eps*n_c) items per class by spectral score, eps = 0.05
  act_clust: per class, flag the smaller 2-means cluster if it holds < 35% of the class
  loss     : top ceil(1.5*eps*n_c) items per class by training loss
Threshold-free AUROC is computed for the two continuous scores within the target class
(label 1, which holds all poison) and over all items."""
import json, math, pathlib, sys
import numpy as np

EPS = 0.05


def auroc(score, pos):
    """Mann-Whitney AUROC with average ranks for ties; None if a class is empty."""
    pos = np.asarray(pos, bool); n1 = int(pos.sum()); n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(score, kind="mergesort"); s = np.asarray(score)[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1; i = j + 1
    r = np.empty(len(s)); r[order] = ranks
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def flags_for(d):
    y, isp = d["label"], d["is_poison"]
    f = {k: np.zeros(len(y), bool) for k in ("spectral", "act_clust", "loss")}
    for c in (0, 1):
        idx = np.flatnonzero(y == c); k = int(math.ceil(1.5 * EPS * len(idx)))
        for name in ("spectral", "loss"):
            sc = d[name][idx]
            top = idx[np.argsort(-sc, kind="stable")[:k]]
            f[name][top] = True
        lab = d["ac_label"][idx]
        small = int((lab == 1).sum() < (lab == 0).sum())
        if (lab == small).mean() < 0.35:
            f["act_clust"][idx[lab == small]] = True
    return f


def metrics(flag, isp):
    tp = int((flag & isp).sum()); fp = int((flag & ~isp).sum()); P = int(isp.sum()); N = len(isp) - P
    fn = P - tp; nf = int(flag.sum())
    rec = tp / P if P else None; prec = tp / nf if nf else None
    f1 = (2 * prec * rec / (prec + rec)) if (rec is not None and prec) else (0.0 if P else None)
    return {"flagged": nf, "tp": tp, "fp": fp, "fn": fn, "recall": rec, "precision": prec, "f1": f1,
            "fpr": fp / N, "fnr": fn / P if P else None}


def main():
    rd = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results_g5")
    g3p = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "results_gpu/g3_results.json")
    checks, problems = [], []
    def check(ok, what):
        checks.append({"check": what, "ok": bool(ok)})
        if not ok: problems.append(what)

    env = json.loads((rd / "environment.json").read_text())
    rows = json.loads((rd / "g5_results.json").read_text())
    smoke = env.get("smoke", False)
    counts = sorted({r["poison"] for r in rows})
    exp_counts = [0, 10, 20] if smoke else [0, 80, 160, 320]
    check(counts == exp_counts and len(rows) == 3 * len(exp_counts), f"all {3*len(exp_counts)} runs present")
    if not smoke:
        check("3070" in env.get("device", ""), "device is the RTX 3070 (" + env.get("device", "?") + ")")
        check(all(r.get("device", "cuda") == "cuda" for r in rows), "every run trained on CUDA")
    check(not any(r.get("deterministic_warn_only") for r in rows), "deterministic kernels throughout")
    check(all(r["receipts_valid"] == r["committed"] for r in rows), "every committed item has a valid receipt")

    per_run = []
    for r in rows:
        tag = f"c{r['poison']}_s{r['seed']}"
        d = dict(np.load(rd / f"scores_{tag}.npz"))
        isp = d["is_poison"].astype(bool)
        n = r["committed"]
        check(all(len(d[k]) == n for k in ("spectral", "ac_label", "loss", "label", "is_poison")), f"{tag}: array lengths = {n}")
        check(int(isp.sum()) == r["poison"], f"{tag}: poison count")
        check(bool((d["label"][isp] == 1).all()), f"{tag}: every poison item labelled positive")
        check(bool(np.isfinite(d["spectral"]).all() and np.isfinite(d["loss"]).all()), f"{tag}: finite scores")
        sizes = r["ac_cluster_sizes"]
        check(all([int((d["ac_label"][d["label"] == c] == j).sum()) for j in (0, 1)] == sizes[str(c)] for c in (0, 1)),
              f"{tag}: cluster sizes match the log")
        fl = flags_for(d)
        out = {"poison": r["poison"], "seed": r["seed"], "committed": n, "clean_acc": r["clean_acc"],
               "attack_success": r["attack_success"], "train_s": r["train_s"], "run_s": r.get("run_s"),
               "gpu_peak_mem_mb": r.get("gpu_peak_mem_mb")}
        for k, f in fl.items():
            out[k] = metrics(f, isp)
        tgt = d["label"] == 1
        for k in ("spectral", "loss"):
            out[k]["auroc_target_class"] = auroc(d[k][tgt], isp[tgt])
            out[k]["auroc_all"] = auroc(d[k], isp)
        per_run.append(out)

    agg = []
    for c in counts:
        rs = [x for x in per_run if x["poison"] == c]
        a = {"poison": c, "poison_rate": c / rs[0]["committed"], "runs": len(rs),
             "clean_acc_mean": float(np.mean([x["clean_acc"] for x in rs])),
             "asr_mean": float(np.mean([x["attack_success"] for x in rs])),
             "asr_min": float(min(x["attack_success"] for x in rs)), "asr_max": float(max(x["attack_success"] for x in rs))}
        for k in ("spectral", "act_clust", "loss"):
            tp = sum(x[k]["tp"] for x in rs); P = sum(x["poison"] for x in rs); nf = sum(x[k]["flagged"] for x in rs)
            a[k] = {"tp_pooled": tp, "poison_pooled": P, "recall_pooled": tp / P if P else None,
                    "precision_pooled": tp / nf if nf else None,
                    "fp_mean": float(np.mean([x[k]["fp"] for x in rs])),
                    "fpr_mean": float(np.mean([x[k]["fpr"] for x in rs])),
                    "runs_flagging_anything": sum(x[k]["flagged"] > 0 for x in rs)}
            if k != "act_clust" and P:
                v = [x[k]["auroc_target_class"] for x in rs]
                a[k]["auroc_target_class_mean"] = float(np.mean(v)); a[k]["auroc_target_class_range"] = [min(v), max(v)]
        agg.append(a)

    replay = None
    if g3p.exists() and not smoke:
        g3 = json.loads(g3p.read_text()); replay = []
        for x in per_run:
            m = [g for g in g3 if g["poison"] == x["poison"] and g["seed"] == x["seed"]]
            if not m: continue
            g = m[0]
            row = {"poison": x["poison"], "seed": x["seed"],
                   "clean_acc_equal": g["clean_acc"] == x["clean_acc"],
                   "asr_equal": g["attack_success"] == x["attack_success"],
                   "g3_asr": g["attack_success"], "g5_asr": x["attack_success"]}
            if g.get("detector_recall") is not None:
                row.update(spectral_recall_equal=abs(g["detector_recall"] - x["spectral"]["recall"]) < 1e-12,
                           spectral_flagged_equal=g["flagged"] == x["spectral"]["flagged"],
                           g3_recall=g["detector_recall"], g5_recall=x["spectral"]["recall"])
            replay.append(row)
        ok = all(all(v for k, v in r.items() if k.endswith("_equal")) for r in replay)
        check(len(replay) == 9 and ok, "G3 runs (0, 80, 160 x seeds 1-3) replayed exactly")

    res = {"environment": env, "validation": {"passed": not problems, "problems": problems, "checks": checks},
           "rules": {"eps": EPS, "act_clust_threshold": 0.35}, "per_run": per_run, "by_poison_level": agg,
           "replay_vs_g3": replay}
    (rd / "g5_analysis.json").write_text(json.dumps(res, indent=1))

    L = [f"# G5 analysis ({rd})", "", f"Device: {env.get('device')} | torch {env.get('torch')} | CUDA {env.get('cuda')}",
         "", f"Validation: {'PASSED' if not problems else 'FAILED'} ({len(checks)} checks)"]
    L += [f"- FAILED: {p}" for p in problems]
    L += ["", "| poison | rate | ASR mean (min-max) | clean acc | spectral recall | spectral FPR | AC recall | AC FPR | AC runs flagging | loss recall | loss FPR | AUROC spectral | AUROC loss |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, p=1: "--" if v is None else f"{100*v:.{p}f}%"
    for a in agg:
        au = lambda k: "--" if "auroc_target_class_mean" not in a[k] else f"{a[k]['auroc_target_class_mean']:.3f}"
        L.append(f"| {a['poison']} | {100*a['poison_rate']:.2f}% | {f(a['asr_mean'])} ({f(a['asr_min'],0)}-{f(a['asr_max'],0)}) | {f(a['clean_acc_mean'])} | "
                 f"{a['spectral']['tp_pooled']}/{a['spectral']['poison_pooled']} | {f(a['spectral']['fpr_mean'])} | "
                 f"{a['act_clust']['tp_pooled']}/{a['act_clust']['poison_pooled']} | {f(a['act_clust']['fpr_mean'])} | {a['act_clust']['runs_flagging_anything']}/3 | "
                 f"{a['loss']['tp_pooled']}/{a['loss']['poison_pooled']} | {f(a['loss']['fpr_mean'])} | {au('spectral')} | {au('loss')} |")
    if replay:
        L += ["", "Replay of G3 runs: " + ("all identical" if all(all(v for k, v in r.items() if k.endswith('_equal')) for r in replay) else "MISMATCH, see g5_analysis.json")]
    (rd / "g5_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if not problems else 2)


if __name__ == "__main__":
    main()
