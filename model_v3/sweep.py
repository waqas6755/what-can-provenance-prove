"""v2 sweep: every mechanism set x pipeline type x goal, for several adversary
configurations, plus targeted assumption checks.  Writes one Alloy program per
adversary configuration, runs them, and stores raw outcomes.
'holds' = Alloy found no counterexample within the scope (bounded result)."""
import itertools, json, subprocess, pathlib, sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
import os
JAR = os.environ.get("ALLOY_JAR", "org.alloytools.alloy.dist.jar")  # Alloy 6.2.0
SCOPE = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MECH = ["BindFix", "BindGrind", "WitSrc", "WitIn", "AttCap", "AttTrans", "Detector"]
ADV_ALL = ["AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec", "AdvWit", "AdvTEE"]
ADVERSARIES = {
    "full":        ["AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec"],
    "full+wit":    ["AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec", "AdvWit"],
    "full+tee":    ["AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec", "AdvTEE"],
    "cap-only":    ["AdvSource", "AdvCap"],
    "exec-only":   ["AdvSource", "AdvExec"],
    "trans-only":  ["AdvSource", "AdvGen", "AdvTrans"],
}
GOALS = ["G_bind", "G_auth", "G_noinf", "G_content", "G_zero", "G_complete"]
base = (HERE / "pipeline_v3_base.als").read_text()

def present(names, universe):
    return " and ".join(("some " if n in names else "no ") + n for n in universe)

def run(adv_name):
    adv = ADVERSARIES[adv_name]
    cmds, meta = [], []
    for gen in (False, True):
        for r in range(len(MECH) + 1):
            for on in itertools.combinations(MECH, r):
                for goal in GOALS:
                    name = f"c{len(cmds)}"
                    cfg = " and ".join([("some Gen" if gen else "no Gen"), "no PerfectDetector",
                                        present(on, MECH), present(adv, ADV_ALL)])
                    cmds.append(f"check {name} {{ ({cfg} and Accepted) implies {goal} }} for {SCOPE}")
                    meta.append({"cmd": name, "adv": adv_name, "gen": gen, "mech": sorted(on), "goal": goal})
    d = HERE / f"run_{adv_name}_s{SCOPE}"
    d.mkdir(exist_ok=True)
    src = d / "checks.als"
    src.write_text(base + "\n-- generated\n" + "\n".join(cmds) + "\n")
    subprocess.run(["java", "-jar", JAR, "exec", "-f", "-t", "json", "-o", str(d / "out"), "-c", "*", str(src)],
                   capture_output=True, text=True)
    rec = json.loads((d / "out" / "receipt.json").read_text())["commands"]
    res = {v["name"]: any(s.get("instances") for s in v.get("solution", [])) for v in rec.values()}
    rows = [{**m, "counterexample": res[m["cmd"]], "holds": not res[m["cmd"]]} for m in meta]
    (d / "results.json").write_text(json.dumps(rows))
    return adv_name, len(rows), len(res)

with ThreadPoolExecutor(max_workers=1) as ex:
    for name, n, got in ex.map(run, ADVERSARIES):
        print(name, "commands", n, "results", got, flush=True)
