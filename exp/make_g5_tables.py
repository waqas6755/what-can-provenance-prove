"""Generate the G5 table rows and macros for the manuscript from the raw G5 analysis.
Usage: python make_g5_tables.py results_g5/g5_analysis.json g3_results.json ../manuscript/gen
Reads only; numbers in the manuscript come from here."""
import json, sys, pathlib
import numpy as np
A = json.load(open(sys.argv[1])); g3 = json.load(open(sys.argv[2])); out = pathlib.Path(sys.argv[3])
env = A["environment"]; runs = A["per_run"]


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]: j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


pct = lambda v, d=0: f"{100 * v:.{d}f}"
rows = []
for a in A["by_poison_level"]:
    c = a["poison"]
    def rec(k):
        return "--" if not c else f"{a[k]['tp_pooled']}/{a[k]['poison_pooled']}"
    def auc(k):
        if not c: return "--"
        lo, hi = a[k]["auroc_target_class_range"]
        return f"{a[k]['auroc_target_class_mean']:.2f} ({lo:.2f}--{hi:.2f})"
    rows.append(" & ".join([str(c), pct(a["poison_rate"], 1) + "\\%",
                            f"{pct(a['asr_mean'])} ({pct(a['asr_min'])}--{pct(a['asr_max'])})",
                            pct(a["clean_acc_mean"], 1), rec("spectral"), auc("spectral"), rec("loss"), auc("loss"),
                            f"{pct(a['act_clust']['fpr_mean'])}\\% ({a['act_clust']['runs_flagging_anything']}/3)"]) + "\\\\")
(out / "tab_g5_rows.tex").write_text("\n".join(rows) + "\n")

rp = A["replay_vs_g3"]
pois = [r for r in runs if r["poison"] > 0]
asr = [r["attack_success"] for r in pois]
diff = [abs(r["g5_asr"] - r["g3_asr"]) for r in rp]
diff_lo = [abs(r["g5_asr"] - r["g3_asr"]) for r in rp if r["poison"] <= 80]
clean_ac = [r["act_clust"]["fpr"] for r in runs if r["poison"] == 0]
lvl = {a["poison"]: a for a in A["by_poison_level"]}
m = {
    "GFiveGPU": env["device"].replace("NVIDIA GeForce ", ""),
    "GFiveMem": f"{env['gpu_total_mem_mb'] / 1024:.0f}",
    "GFiveTorch": env["torch"].split("+")[0], "GFiveCuda": env["cuda"],
    "GFivePy": env["python"].split()[0], "GFiveTf": env["transformers"],
    "GFiveRuns": str(len(runs)),
    "GFiveTrainS": f"{np.mean([r['train_s'] for r in runs]):.0f}",
    "GFivePeakGB": f"{max(r['gpu_peak_mem_mb'] for r in runs) / 1024:.1f}",
    "GFiveChecksOK": str(sum(c["ok"] for c in A["validation"]["checks"])),
    "GFiveChecks": str(len(A["validation"]["checks"])),
    "GFiveReplayN": str(len(rp)),
    "GFiveReplayAsrEq": str(sum(r["asr_equal"] for r in rp)),
    "GFiveReplayAccEq": str(sum(r["clean_acc_equal"] for r in rp)),
    "GFiveReplayMaxDiff": f"{100 * max(diff):.1f}",
    "GFiveReplayLowMaxDiff": f"{100 * max(diff_lo):.1f}",
    "GFiveRhoSpec": f"{spearman(asr, [r['spectral']['recall'] for r in pois]):.2f}",
    "GFiveRhoLoss": f"{spearman(asr, [r['loss']['recall'] for r in pois]):.2f}",
    "GFiveACCleanMin": pct(min(clean_ac)), "GFiveACCleanMax": pct(max(clean_ac)),
    "GFiveACPrecMax": pct(max(r["act_clust"]["precision"] for r in pois), 1),
    "GFiveSpecAucLow": f"{lvl[80]['spectral']['auroc_target_class_mean']:.2f}",
    "GFiveLossAucLow": f"{lvl[80]['loss']['auroc_target_class_mean']:.2f}",
    "GFiveSpecAucHigh": f"{lvl[320]['spectral']['auroc_target_class_mean']:.2f}",
    "GFiveLossAucHigh": f"{lvl[320]['loss']['auroc_target_class_mean']:.2f}",
    "GFiveSpecRecLow": pct(lvl[80]["spectral"]["recall_pooled"]),
    "GFiveLossRecLow": pct(lvl[80]["loss"]["recall_pooled"]),
    "GFiveSpecPrecLow": pct(lvl[80]["spectral"]["precision_pooled"], 1),
    "GFiveRecHighTP": str(lvl[320]["spectral"]["tp_pooled"] + 0),
    "GFiveLossRecHighTP": str(lvl[320]["loss"]["tp_pooled"]),
    "GFiveHighN": str(lvl[320]["spectral"]["poison_pooled"]),
    "GFiveAsrHighMin": pct(lvl[320]["asr_min"], 1),
}
(out / "numbers_g5.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
print(json.dumps(m, indent=1))
