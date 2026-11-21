"""Revision R1: decide the model for EVERY subset of the seven adversary capabilities
(2^7 = 128 adversaries), not only the six named profiles.

Vectorised re-implementation of the semantics in model_v3/independent_check.py.
It is validated first against all 9,216 Alloy results of the six named profiles,
then run over the full lattice: 128 adversaries x 128 mechanism sets x 2 pipeline
types x 6 goals = 196,608 decisions.  Exact by the separability lemma.

Outputs: lattice_results.json (holds/fails per instance), lattice_summary.json
(minimal sufficient mechanism sets per adversary/pipeline/goal, monotonicity checks)."""
import itertools, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
V3 = pathlib.Path(__file__).resolve().parent.parent / "model_v3"
REL = ["delivered", "synth", "filtered", "committed", "used", "audSynth", "audD", "audF", "bad", "missed", "examined"]
MECH = ["BindFix", "BindGrind", "WitSrc", "WitIn", "AttCap", "AttTrans", "Detector"]
ADV = ["AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec", "AdvWit", "AdvTEE"]
GOALS = ["G_bind", "G_auth", "G_noinf", "G_content", "G_zero", "G_complete"]
NAMED = {
    "full": {"AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec"},
    "full+wit": {"AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec", "AdvWit"},
    "full+tee": {"AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec", "AdvTEE"},
    "cap-only": {"AdvSource", "AdvCap"},
    "exec-only": {"AdvSource", "AdvExec"},
    "trans-only": {"AdvSource", "AdvGen", "AdvTrans"},
}
B = np.array(list(itertools.product((False, True), repeat=len(REL))), dtype=bool)   # 2048 x 11
col = {r: B[:, i] for i, r in enumerate(REL)}
d, s, f, c, u = col["delivered"], col["synth"], col["filtered"], col["committed"], col["used"]
aS, aD, aF, bad, mis, ex = col["audSynth"], col["audD"], col["audF"], col["bad"], col["missed"], col["examined"]
imp = lambda a, b: (~a) | b
T = np.ones(len(B), dtype=bool)


def holds(g, goal):
    """g: set of names (adversary capabilities, mechanisms, 'Gen'). Returns True if goal is achieved."""
    tee = "AdvTEE" not in g
    bound = "BindFix" in g or "BindGrind" in g
    ok = (~(d & s))
    if "Gen" not in g: ok &= ~s & ~aS
    ok &= imp(mis, bad)
    if "AdvSource" not in g: ok &= ~(bad & d)
    if "AdvGen" not in g: ok &= ~(bad & s)
    if "AdvTrans" not in g or ("AttTrans" in g and tee): ok &= imp(f, d | s)
    if "AttTrans" in g and tee: ok &= (aS == s)
    if "AdvTrans" not in g and "AdvCap" not in g: ok &= (aS == s)
    if "AdvCap" not in g or ("AttCap" in g and tee): ok &= (c == f)
    if "AdvExec" not in g: ok &= (u == f)
    if "AdvWit" not in g: ok &= (aD == d) & (aF == f)
    if "Detector" in g and tee: ok &= (ex == c)
    else: ok &= ~ex
    acc = T.copy()
    if bound: acc &= (u == c)
    if "WitSrc" in g: acc &= imp(c, aD | aS)
    if "WitIn" in g: acc &= imp(aF, c)
    if "Detector" in g and tee: acc &= imp(c & bad, mis)
    goalv = {"G_bind": u == c, "G_noinf": u == c, "G_auth": imp(u, d | s),
             "G_content": imp(u & bad, mis & ex), "G_zero": ~(u & bad), "G_complete": imp(f, c)}[goal]
    assert ok[0] and acc[0] and goalv[0]          # separability side condition: zero vector admissible
    if goal == "G_noinf" and "AdvExec" in g and "BindFix" not in g:
        return False                               # ground seed admissible: global violation
    return not np.any(ok & acc & ~goalv)


def subsets(xs):
    return [frozenset(c) for r in range(len(xs) + 1) for c in itertools.combinations(xs, r)]


MSETS = subsets(MECH)

if __name__ == "__main__":
    # 1) validate against the Alloy sweep of the six named profiles
    agree = 0; total = 0
    for name, adv in NAMED.items():
        for r in json.loads((V3 / f"run_{name}_s5" / "results.json").read_text()):
            g = set(adv) | set(r["mech"]) | ({"Gen"} if r["gen"] else set())
            total += 1; agree += (holds(g, r["goal"]) == r["holds"])
    print("validation vs Alloy:", agree, "/", total, flush=True)
    assert agree == total
    # 2) full lattice
    res = {}
    for adv in subsets(ADV):
        for gen in (False, True):
            for m in MSETS:
                g = set(adv) | set(m) | ({"Gen"} if gen else set())
                for goal in GOALS:
                    res[(adv, gen, m, goal)] = holds(g, goal)
    print("lattice decisions:", len(res), flush=True)
    # monotonicity in mechanisms (Prop. 1) and anti-monotonicity in adversary capabilities
    mono_mech = all(res[(a, gn, m2, gl)] for (a, gn, m, gl), v in res.items() if v
                    for x in MECH if x not in m for m2 in [m | {x}])
    anti_adv = all(res[(a, gn, m, gl)] for (a2, gn, m, gl), v in res.items() if v
                   for a in [a2 - {x} for x in a2])
    # minimal sufficient sets per (adversary, gen, goal)
    summ = {}
    for adv in subsets(ADV):
        for gen in (False, True):
            for goal in GOALS:
                ok = [m for m in MSETS if res[(adv, gen, m, goal)]]
                minimal = [sorted(m) for m in ok if not any(o < m for o in ok)]
                summ[f"{'+'.join(sorted(adv)) or 'none'}|{'gen' if gen else 'filter'}|{goal}"] = minimal
    unreachable = {k: v for k, v in summ.items() if not v}
    out = {"validated_against_alloy": f"{agree}/{total}", "decisions": len(res),
           "monotone_in_mechanisms": mono_mech, "antimonotone_in_adversary": anti_adv,
           "n_unreachable_cases": len(unreachable)}
    (HERE / "lattice_summary.json").write_text(json.dumps({**out, "minimal_sets": summ}, indent=0))
    (HERE / "lattice_results.json").write_text(json.dumps(
        [{"adv": sorted(a), "gen": gn, "mech": sorted(m), "goal": gl, "holds": v} for (a, gn, m, gl), v in res.items()]))
    print(json.dumps(out, indent=1))
