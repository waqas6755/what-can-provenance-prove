"""Regenerate paper/gen/tab_e2d_rows.tex (Table VIII, E2d) from results_revision/e2d_results.json.
Per poison level, over the 3 seeds: mean attack success; detected poison summed over seeds /
poison items summed over seeds; clean false positives as the mean count per run and the mean
false-positive rate."""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
rows = json.loads((ROOT / "results_revision" / "e2d_results.json").read_text())
levels = sorted({r["poison"] for r in rows})
out = []
for p in levels:
    rs = [r for r in rows if r["poison"] == p]
    asr = sum(r["attack_success"] for r in rs) / len(rs)
    cells = [str(p), f"{round(100 * asr)}\\%"]
    for d in ("spectral", "act_clust", "loss"):
        tp = sum(r[d]["tp"] for r in rs); n = sum(r["poison"] for r in rs)
        fp = sum(r[d]["fp"] for r in rs) / len(rs); fpr = sum(r[d]["fpr"] for r in rs) / len(rs)
        cells.append("--" if p == 0 else f"{tp}/{n}")
        cells.append(f"{round(fp)} ({100 * fpr:.1f}\\%)")
    out.append(" & ".join(cells) + "\\\\")
(ROOT / "paper" / "gen" / "tab_e2d_rows.tex").write_text("\n".join(out) + "\n")
