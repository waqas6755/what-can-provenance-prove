"""E2: run the prototype for every adversary profile x pipeline type x mechanism set,
let the adversary try every combination of its available actions and poison budgets,
and record, per goal, whether some ACCEPTED execution violates the goal.
Compare with the Alloy v3 sweep (model_v3/run_<adv>_s5/results.json)."""
import itertools, json, pathlib, sys, time
import pipeline as P

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "e2"
OUT.mkdir(parents=True, exist_ok=True)
ALLOY = pathlib.Path(__file__).resolve().parent.parent / "model_v3"


def sweep(adv_name):
    adv = frozenset(P.ADVERSARIES[adv_name])
    rows = []
    for gen in (False, True):
        acts_av = P.available(adv, gen)
        plans = [frozenset(c) for r in range(len(acts_av) + 1) for c in itertools.combinations(acts_av, r)]
        for r in range(len(P.MECH) + 1):
            for on in itertools.combinations(P.MECH, r):
                mech = frozenset(on)
                viol = {g: None for g in P.GOALS}
                honest_ok = P.run(adv, gen, mech, frozenset(), 1)[0]
                n_acc = n_tried = 0
                for plan in plans:
                    if all(v is not None for v in viol.values()):
                        break                      # every goal already violated: nothing left to show
                    for b in P.BUDGETS:
                        if not any(a in plan for a in ("src_poison", "gen_poison", "trans_inject", "cap_inject", "exec_inject")) and b != 1:
                            continue
                        n_tried += 1
                        ok, g, _ = P.run(adv, gen, mech, plan, b)
                        if not ok:
                            continue
                        n_acc += 1
                        for k, v in g.items():
                            if not v and viol[k] is None:
                                viol[k] = {"plan": sorted(plan), "budget": b}
                for k in P.GOALS:
                    rows.append({"adv": adv_name, "gen": gen, "mech": sorted(mech), "goal": k,
                                 "violated": viol[k] is not None, "witness": viol[k],
                                 "honest_accepted": honest_ok, "accepted_plans": n_acc, "executions_tried": n_tried})
    return rows


def compare(rows, adv_name):
    alloy = json.loads((ALLOY / f"run_{adv_name}_s5" / "results.json").read_text())
    key = lambda r: (r["gen"], tuple(sorted(r["mech"])), r["goal"])
    A = {key(r): r["holds"] for r in alloy}
    stats = {"sound_ok": 0, "unsound": [], "tight": 0, "not_realised": []}
    for r in rows:
        holds = A[key(r)]
        if holds:
            if r["violated"]:
                stats["unsound"].append(key(r))
            else:
                stats["sound_ok"] += 1
        else:
            if r["violated"]:
                stats["tight"] += 1
            else:
                stats["not_realised"].append(key(r))
    return stats


if __name__ == "__main__":
    advs = sys.argv[1:] or list(P.ADVERSARIES)
    for a in advs:
        t0 = time.time()
        rows = sweep(a)
        st = compare(rows, a)
        (OUT / f"{a}.json").write_text(json.dumps({"rows": rows, "compare": {
            "model_holds_and_no_violation": st["sound_ok"], "model_holds_but_violated": st["unsound"],
            "model_fails_and_violation_realised": st["tight"], "model_fails_not_realised": st["not_realised"]}},
            default=list))
        print(a, len(rows), "sound", st["sound_ok"], "UNSOUND", len(st["unsound"]),
              "tight", st["tight"], "not_realised", len(st["not_realised"]), f"{time.time()-t0:.0f}s", flush=True)
