"""Revision R2: check each theorem statement, as written in the manuscript, against all
196,608 lattice decisions (every adversary subset), not only the six named profiles."""
import json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
rows = json.loads((HERE / "lattice_results.json").read_text())
R = {(frozenset(r["adv"]), r["gen"], frozenset(r["mech"]), r["goal"]): r["holds"] for r in rows}
report = {}

def check(name, pred, cond):
    """pred(adv, gen, mech, goal) -> predicted holds; cond -> instance in theorem's scope."""
    n = bad = 0; ex = []
    for (a, gn, m, gl), v in R.items():
        if not cond(a, gn, m, gl): continue
        n += 1
        if pred(a, gn, m, gl) != v:
            bad += 1
            if len(ex) < 3: ex.append({"adv": sorted(a), "gen": gn, "mech": sorted(m), "goal": gl, "holds": v})
    report[name] = {"instances_in_scope": n, "mismatches": bad, "examples": ex}

tau = lambda a: "AdvTEE" not in a
beta = lambda m: "BindFix" in m or "BindGrind" in m
# Theorem 1 (Binding)
check("Thm1_binding",
      lambda a, gn, m, gl: beta(m) if "AdvExec" in a else ("AdvCap" not in a or ("AttCap" in m and tau(a)) or beta(m)),
      lambda a, gn, m, gl: gl == "G_bind")
# Theorem 2 (Lineage, full adversary): exact formula for profile full
FULL = frozenset({"AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec"})
check("Thm2_lineage_full",
      lambda a, gn, m, gl: beta(m) and (("WitSrc" in m and (not gn or "AttTrans" in m)) or ("AttCap" in m and "AttTrans" in m)),
      lambda a, gn, m, gl: gl == "G_auth" and a == FULL)
# Theorem 3 (Influence)
check("Thm3_influence", lambda a, gn, m, gl: "BindFix" in m,
      lambda a, gn, m, gl: gl == "G_noinf" and "AdvExec" in a)
# Theorem 5 part 1 (no G_zero) with AdvSource
check("Thm5_part1_zero_unreachable", lambda a, gn, m, gl: False,
      lambda a, gn, m, gl: gl == "G_zero" and "AdvSource" in a)
# Theorem 5 part 2
check("Thm5_part2_content_untrusted_exec", lambda a, gn, m, gl: beta(m) and "Detector" in m,
      lambda a, gn, m, gl: gl == "G_content" and "AdvSource" in a and tau(a) and "AdvExec" in a)
# Theorem 5 part 3 (Detector and G_compl achieved)
compl = lambda a, gn, m: R[(a, gn, m, "G_complete")]
check("Thm5_part3_content_trusted_exec", lambda a, gn, m, gl: "Detector" in m and compl(a, gn, m),
      lambda a, gn, m, gl: gl == "G_content" and "AdvSource" in a and tau(a) and "AdvExec" not in a and "AdvWit" not in a)
# Theorem 5 broken attestation
check("Thm5_tee_broken", lambda a, gn, m, gl: False,
      lambda a, gn, m, gl: gl == "G_content" and "AdvSource" in a and not tau(a))
# Theorem 6 (Completeness)
check("Thm6_completeness",
      lambda a, gn, m, gl: ("AttCap" in m and tau(a)) or ("WitIn" in m and "AdvWit" not in a) or (beta(m) and "AdvExec" not in a),
      lambda a, gn, m, gl: gl == "G_complete" and "AdvCap" in a)
(HERE / "theorem_lattice_check.json").write_text(json.dumps(report, indent=1))
for k, v in report.items(): print(k, v["instances_in_scope"], "mismatches:", v["mismatches"], v["examples"][:1])
