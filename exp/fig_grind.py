"""Figure: seed grinding. Left: exact success probability vs T on the prototype (E3) with the
bound T/2^k. Right: bits set with probability >= 1/2 for the prototype, DistilBERT (G1) and
Qwen2.5-0.5B + LoRA (G4), with the bound log2 T + 1."""
import json, math, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parent.parent / "results"
out = sys.argv[1]
e3 = json.loads((R / "e3" / "e3_exact.json").read_text())
S = {tuple(map(int, k.split(","))): v for k, v in e3["success"].items()}
Ts = sorted({T for _, T in S})
plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})  # TrueType: no Type 3 fonts in the PDF
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
cmap = plt.get_cmap("viridis")
for i, k in enumerate([1, 2, 4, 6, 8, 10]):
    c = cmap(i / 5)
    ax[0].plot(Ts, [S[(k, T)] for T in Ts], "o-", ms=2.5, lw=1, color=c, label=f"k={k}")
    ax[0].plot(Ts, [min(1, T / 2 ** k) for T in Ts], ":", lw=0.8, color=c)
ax[0].set_xscale("log", base=2); ax[0].set_xlabel(r"seeds tried by the prover, $T$")
ax[0].set_ylabel(r"Pr[$k$-bit target realised]"); ax[0].legend(frameon=False, fontsize=6, ncol=2)
ax[0].set_title(r"prototype: exact (solid) vs bound $T/2^k$ (dotted)", fontsize=7)
series = [("prototype MLP (1,024 runs)", R / "e3" / "e3_exact.json", "o-"),
          ("DistilBERT (160 runs)", R / "gpu" / "g1_exact.json", "s-"),
          ("Qwen2.5-0.5B + LoRA (48 runs)", R / "gpu" / "g4_exact.json", "^-")]
allT = set()
for lab, f, st in series:
    d = json.loads(f.read_text()); t = sorted(map(int, d["bits_at_half"])); allT |= set(t)
    ax[1].plot([math.log2(x) for x in t], [d["bits_at_half"][str(x)] for x in t], st, ms=3, lw=1, label=lab)
t = sorted(allT)
ax[1].plot([math.log2(x) for x in t], [math.log2(x) + 1 for x in t], "--", lw=0.8, color="gray", label=r"bound $\log_2 T + 1$")
ax[1].set_xlabel(r"$\log_2 T$"); ax[1].set_ylabel(r"bits set with prob. $\geq 1/2$"); ax[1].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("ok", out)
