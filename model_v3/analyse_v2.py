"""Minimal sufficient mechanism sets per adversary, pipeline type and goal, plus
the maximal failing sets (the cases a minimality proof must break)."""
import json, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
scope = sys.argv[1] if len(sys.argv) > 1 else "5"
out = {}
for d in sorted(HERE.glob(f"run_*_s{scope}")):
    rows = json.loads((d / "results.json").read_text())
    for r in rows:
        out.setdefault((r["adv"], r["gen"], r["goal"]), []).append((frozenset(r["mech"]), r["holds"]))
report = []
for (adv, gen, goal), lst in sorted(out.items(), key=lambda k: (k[0][0], k[0][1], k[0][2])):
    ok = [m for m, h in lst if h]; bad = [m for m, h in lst if not h]
    minimal = sorted([sorted(m) for m in ok if not any(o < m for o in ok)])
    maxfail = sorted([sorted(m) for m in bad if not any(b > m for b in bad)])
    upward = all(h == any(frozenset(x) <= m for x in minimal) for m, h in lst)
    report.append({"adv": adv, "gen": gen, "goal": goal, "holds_in": len(ok), "of": len(lst),
                   "minimal_sufficient": minimal, "maximal_failing": maxfail, "upward_closed": upward})
(HERE / f"summary_s{scope}.json").write_text(json.dumps(report, indent=1))
for r in report:
    if True:
        print(f"{r['adv']:10s} {'gen ' if r['gen'] else 'filt'} {r['goal']:9s} {r['holds_in']:3d}/128 "
              f"min={r['minimal_sufficient'] or 'NONE'} up={r['upward_closed']}")
