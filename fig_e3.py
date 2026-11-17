import json, math, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); kmax = int(sys.argv[3]) if len(sys.argv) > 3 else 10
r = json.loads(src.read_text())
S = {tuple(map(int, k.split(","))): v for k, v in r["success"].items()}
Ts = sorted({T for _, T in S}); ks = sorted({k for k, _ in S})
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
cmap = plt.get_cmap("viridis")
for i, k in enumerate([k for k in ks if k in (1, 2, 4, 6, 8, 10)]):
    c = cmap(i / 5)
    ax[0].plot(Ts, [S[(k, T)] for T in Ts], "o-", ms=2.5, lw=1, color=c, label=f"k={k}")
    ax[0].plot(Ts, [min(1, T / 2 ** k) for T in Ts], ":", lw=0.8, color=c)
ax[0].set_xscale("log", base=2); ax[0].set_xlabel(r"seeds tried by the prover, $T$"); ax[0].set_ylabel(r"Pr[$k$-bit target realised]")
ax[0].legend(frameon=False, fontsize=6, ncol=2); ax[0].set_title(r"measured (solid) vs bound $T/2^k$ (dotted)", fontsize=7)
bits = [r["bits_at_half"][str(T)] for T in Ts]
ax[1].plot([math.log2(T) for T in Ts], bits, "o-", ms=3, label=r"measured (success $\geq 1/2$)")
ax[1].plot([math.log2(T) for T in Ts], [math.log2(T) + 1 for T in Ts], "--", lw=0.8, color="gray", label=r"bound: $k \leq \log_2 T + 1$")
ax[1].set_xlabel(r"$\log_2 T$"); ax[1].set_ylabel(r"bits set with prob. $\geq 1/2$"); ax[1].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("ok", out)
