"""Figure: per-run detector AUROC against attack success (G5). Reads only the raw analysis.
Usage: python fig_g5.py results_g5/g5_analysis.json ../manuscript/fig/g5_detection.pdf"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
A = json.load(open(sys.argv[1]))
runs = [r for r in A["per_run"] if r["poison"] > 0]
style = {80: ("#86b6ef", "o", "1.0% poison"), 160: ("#2a78d6", "s", "2.0% poison"), 320: ("#104281", "^", "3.8% poison")}
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
for ax, key, title in zip(axes, ("spectral", "loss"), ("Spectral signatures", "Training loss")):
    ax.axhline(0.5, color="#8a8984", lw=0.8, ls=(0, (3, 2)), zorder=1)
    for c, (col, mk, lab) in style.items():
        pts = [(100 * r["attack_success"], r[key]["auroc_target_class"]) for r in runs if r["poison"] == c]
        ax.scatter(*zip(*pts), s=22, marker=mk, color=col, edgecolor="#fcfcfb", linewidth=0.8, zorder=3, label=lab)
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_xlim(0, 105); ax.set_ylim(0.2, 1.0); ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("Attack success (%)")
axes[0].set_ylabel("AUROC (poison vs. clean)")
axes[0].text(2, 0.52, "chance", color="#52514e", fontsize=6.5, va="bottom")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.55, 1.03), handletextpad=0.2, columnspacing=0.9)
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6)
fig.savefig(sys.argv[2], bbox_inches="tight", pad_inches=0.02)
