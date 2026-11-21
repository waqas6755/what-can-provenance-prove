"""Revision R3: statistical re-analysis of existing raw results (raw files are read only).
- Wilson 95% intervals for every binomial proportion, with its n.
- Per-seed values; baseline-adjusted attack success (ASR(poison) - ASR(clean), same seed).
- Detector false-positive counts and rates.
- Recount of E2 executions from the raw sweep files.
Writes results_revision/*.json and LaTeX table rows for the manuscript and supplement."""
import json, math, pathlib, statistics as st
V3 = pathlib.Path(__file__).resolve().parent.parent / "results"
OUT = pathlib.Path(__file__).resolve().parent.parent / "results_revision"; OUT.mkdir(exist_ok=True)
GEN = pathlib.Path(__file__).resolve().parent.parent / "paper" / "gen"


def wilson(k, n, z=1.959964):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def kn(p, n):
    k = round(p * n); assert abs(k / n - p) < 1e-9, (p, n); return k


pct = lambda x: f"{100 * x:.1f}"
res = {}

# ---------------- E2 recount
tot_runs = tot_acc = 0; per = {}
for f in sorted((V3 / "e2").glob("*.json")):
    d = json.loads(f.read_text()); seen = {}
    for r in d["rows"]:
        seen[(r["gen"], tuple(r["mech"]))] = (r["executions_tried"], r["accepted_plans"])
    runs = sum(v[0] for v in seen.values()); acc = sum(v[1] for v in seen.values())
    per[f.stem] = {"executions": runs, "accepted": acc, "instances": len(d["rows"]),
                   "unsound": len(d["compare"]["model_holds_but_violated"]),
                   "realised": d["compare"]["model_fails_and_violation_realised"],
                   "not_realised": len(d["compare"]["model_fails_not_realised"])}
    tot_runs += runs; tot_acc += acc
res["E2_recount"] = {"per_profile": per, "total_executions": tot_runs, "total_accepted": tot_acc}

# ---------------- G3 (DistilBERT, SST-2): n_val = 872, n_val_neg = 428 (verified below)
g3 = json.loads((V3 / "gpu" / "g3_results.json").read_text())
NVAL, NNEG = 872, 428
base = {r["seed"]: r["attack_success"] for r in g3 if r["poison"] == 0}
rows = []
for r in g3:
    k_asr = kn(r["attack_success"], NNEG); k_acc = kn(r["clean_acc"], NVAL)
    row = {"poison": r["poison"], "seed": r["seed"], "asr": r["attack_success"], "asr_k": k_asr, "asr_n": NNEG,
           "asr_ci": wilson(k_asr, NNEG), "asr_minus_clean": r["attack_success"] - base[r["seed"]],
           "clean_acc": r["clean_acc"], "acc_ci": wilson(k_acc, NVAL)}
    if r["poison"]:
        tp = kn(r["detector_recall"], r["poison"]); fp = r["flagged"] - tp; clean = r["committed"] - r["poison"]
        row.update(recall=r["detector_recall"], tp=tp, recall_ci=wilson(tp, r["poison"]), flagged=r["flagged"],
                   fp=fp, fpr=fp / clean, fpr_ci=wilson(fp, clean), precision=tp / r["flagged"])
    rows.append(row)
res["G3_per_seed"] = rows
summ = []
for n in sorted({r["poison"] for r in rows}):
    g = [r for r in rows if r["poison"] == n]
    s = {"poison": n, "seeds": len(g), "asr_mean": st.mean(r["asr"] for r in g), "asr_sd": st.stdev(r["asr"] for r in g),
         "adj_mean": st.mean(r["asr_minus_clean"] for r in g), "adj_min": min(r["asr_minus_clean"] for r in g),
         "adj_max": max(r["asr_minus_clean"] for r in g), "acc_mean": st.mean(r["clean_acc"] for r in g)}
    if n:
        s.update(recall_mean=st.mean(r["recall"] for r in g), recall_sd=st.stdev(r["recall"] for r in g),
                 fp_mean=st.mean(r["fp"] for r in g), fpr_mean=st.mean(r["fpr"] for r in g),
                 precision_mean=st.mean(r["precision"] for r in g))
    summ.append(s)
res["G3_summary"] = summ

# ---------------- E2c (prototype): n_eval_neg = 100, clean-acc set = 300
e2c = json.loads((V3 / "e2c" / "e2c_results.json").read_text())
base_c = {r["seed"]: r["attack_success"] for r in e2c if r["poison"] == 0}
rc = []
for r in e2c:
    k = kn(r["attack_success"], 100)
    row = {"poison": r["poison"], "seed": r["seed"], "asr": r["attack_success"], "asr_ci": wilson(k, 100),
           "asr_minus_clean": r["attack_success"] - base_c[r["seed"]], "clean_acc": r["clean_acc"], "flagged": r["flagged"]}
    if r["poison"]:
        tp = r["poison_flagged"]; clean = 1500 - 0 + 0  # committed = 1,500 source items + poison
        row.update(recall=tp / r["poison"], tp=tp, recall_ci=wilson(tp, r["poison"]), fp=r["flagged"] - tp,
                   fpr=(r["flagged"] - tp) / 1500)
    rc.append(row)
res["E2c_per_seed"] = rc

# ---------------- grinding: exact analyses already computed; collect
for nm, f in (("E3", V3 / "e3" / "e3_exact.json"), ("G1", V3 / "gpu" / "g1_exact.json"), ("G4", V3 / "gpu" / "g4_exact.json")):
    d = json.loads(f.read_text())
    res[f"{nm}_grinding"] = {"N_eval": d["N_eval"], "N_selection": d["N_selection"], "bits_at_half": d["bits_at_half"]}

(OUT / "stats_revision.json").write_text(json.dumps(res, indent=1))

# ---------------- LaTeX rows
L = []
for s in summ:
    g = [r for r in rows if r["poison"] == s["poison"]]
    asr_seeds = "; ".join(f"{pct(r['asr'])} [{pct(r['asr_ci'][0])},{pct(r['asr_ci'][1])}]" for r in g)
    if s["poison"]:
        rec = "; ".join(f"{pct(r['recall'])} [{pct(r['recall_ci'][0])},{pct(r['recall_ci'][1])}]" for r in g)
        fp = f"{s['fp_mean']:.0f} ({100 * s['fpr_mean']:.1f}\\%)"
    else:
        rec, fp = "--", "--"
    L.append(f"{s['poison']} & {asr_seeds} & {100 * s['adj_mean']:+.1f} & {rec} & {fp}\\\\")
(GEN / "tab_g3_stats_rows.tex").write_text("\n".join(L) + "\n")
mac = [f"\\newcommand{{\\RevEtwoRuns}}{{{tot_runs:,}}}".replace(",", "{,}"),
       f"\\newcommand{{\\RevEtwoAcc}}{{{tot_acc:,}}}".replace(",", "{,}")]
s160 = [s for s in summ if s["poison"] == 160][0]; s20 = [s for s in summ if s["poison"] == 20][0]
mac += [f"\\newcommand{{\\GthreeAdjHigh}}{{{100 * s160['adj_mean']:.1f}}}",
        f"\\newcommand{{\\GthreeAdjHighMin}}{{{100 * s160['adj_min']:.1f}}}",
        f"\\newcommand{{\\GthreeAdjHighMax}}{{{100 * s160['adj_max']:.1f}}}",
        f"\\newcommand{{\\GthreeFPHigh}}{{{s160['fp_mean']:.0f}}}",
        f"\\newcommand{{\\GthreePrecHigh}}{{{100 * s160['precision_mean']:.1f}}}",
        f"\\newcommand{{\\GthreePrecLow}}{{{100 * s20['precision_mean']:.1f}}}",
        f"\\newcommand{{\\GthreeFPRHigh}}{{{100 * s160['fpr_mean']:.1f}}}"]
(GEN / "numbers_revision.tex").write_text("\n".join(mac) + "\n")
print(json.dumps({"E2": {"runs": tot_runs, "acc": tot_acc}, "G3": summ}, indent=1))
