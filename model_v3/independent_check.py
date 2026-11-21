"""Independent re-encoding of the v2/v3 semantics in plain Python, written from the
prose definitions (report section 'v2 semantics'), not translated from the .als.

Separability lemma (see report): every fact, acceptance condition and goal is a
universal per-item constraint over the global booleans, and the all-false item
vector satisfies every constraint.  Hence a counterexample exists in some finite
universe iff one exists over a single item, so enumerating the 2^11 membership
vectors of one item, plus the global booleans, is an exact (unbounded) check.
The script also verifies the lemma's side condition (all-false vector admissible)
and compares every outcome with the Alloy sweep."""
import itertools, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
REL = ["delivered", "synth", "filtered", "committed", "used", "audSynth", "audD", "audF", "bad", "missed", "examined"]
MECH = ["BindFix", "BindGrind", "WitSrc", "WitIn", "AttCap", "AttTrans", "Detector"]
ADVERSARIES = {
    "full": {"AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec"},
    "full+wit": {"AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec", "AdvWit"},
    "full+tee": {"AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec", "AdvTEE"},
    "cap-only": {"AdvSource", "AdvCap"},
    "exec-only": {"AdvSource", "AdvExec"},
    "trans-only": {"AdvSource", "AdvGen", "AdvTrans"},
}
GOALS = ["G_bind", "G_auth", "G_noinf", "G_content", "G_zero", "G_complete"]

def imp(a, b): return (not a) or b

def item_ok(v, g):
    """Per-item admissibility: domain facts, stage semantics, trust assumptions."""
    tee = "AdvTEE" not in g
    return all([
        not (v["delivered"] and v["synth"]),                                  # sources and generator disjoint
        imp("Gen" not in g, not v["synth"] and not v["audSynth"]),           # declared stage types
        imp(v["missed"], v["bad"]),                                           # misses are poisoned items
        imp("PerfectDetector" in g, not v["missed"]),
        imp("AdvSource" not in g, not (v["bad"] and v["delivered"])),        # honest sources are clean
        imp("AdvGen" not in g, not (v["bad"] and v["synth"])),               # honest generator is clean
        imp("AdvTrans" not in g or ("AttTrans" in g and tee), imp(v["filtered"], v["delivered"] or v["synth"])),
        imp("AttTrans" in g and tee, v["audSynth"] == v["synth"]),           # attested transform reports truthfully
        imp("AdvTrans" not in g and "AdvCap" not in g, v["audSynth"] == v["synth"]),
        imp("AdvCap" not in g or ("AttCap" in g and tee), v["committed"] == v["filtered"]),
        imp("AdvExec" not in g, v["used"] == v["filtered"]),                 # honest executor trains on its input
        imp("AdvWit" not in g, v["audD"] == v["delivered"] and v["audF"] == v["filtered"]),
        imp("Detector" in g and tee, v["examined"] == v["committed"]),
        imp("Detector" not in g or not tee, not v["examined"]),
    ])

def item_accepted(v, g):
    tee = "AdvTEE" not in g
    bound = "BindFix" in g or "BindGrind" in g
    return all([
        imp(bound, v["used"] == v["committed"]),
        imp("WitSrc" in g, imp(v["committed"], v["audD"] or v["audSynth"])),
        imp("WitIn" in g, imp(v["audF"], v["committed"])),
        imp("Detector" in g and tee, imp(v["committed"] and v["bad"], v["missed"])),
    ])

def item_goal(v, goal):
    return {
        "G_bind": v["used"] == v["committed"],
        "G_auth": imp(v["used"], v["delivered"] or v["synth"]),
        "G_noinf": v["used"] == v["committed"],
        "G_content": imp(v["used"] and v["bad"], v["missed"] and v["examined"]),
        "G_zero": not (v["used"] and v["bad"]),
        "G_complete": imp(v["filtered"], v["committed"]),
    }[goal]

def global_ok(g, ground):
    # GroundSeed only if the executor is corrupt and the seed is not fixed independently of the data
    return imp(ground, "AdvExec" in g and "BindFix" not in g)

VECTORS = [dict(zip(REL, bits)) for bits in itertools.product((False, True), repeat=len(REL))]
ZERO = dict.fromkeys(REL, False)

def violation(g, goal):
    for ground in (False, True):
        if not global_ok(g, ground):
            continue
        assert item_ok(ZERO, g) and item_accepted(ZERO, g) and item_goal(ZERO, goal)  # lemma side condition
        if goal == "G_noinf" and ground:
            return True          # global violation: the zero vector is an accepted instance
        for v in VECTORS:
            if item_ok(v, g) and item_accepted(v, g) and not item_goal(v, goal):
                return True
    return False

if __name__ == "__main__":
    scope = sys.argv[1] if len(sys.argv) > 1 else "5"
    agree = disagree = 0; diffs = []
    for adv, advset in ADVERSARIES.items():
        rows = json.loads((HERE / f"run_{adv}_s{scope}" / "results.json").read_text())
        for r in rows:
            g = set(advset) | set(r["mech"]) | ({"Gen"} if r["gen"] else set())
            mine_holds = not violation(g, r["goal"])
            if mine_holds == r["holds"]:
                agree += 1
            else:
                disagree += 1; diffs.append(r)
    print(json.dumps({"compared": agree + disagree, "agree": agree, "disagree": disagree,
                      "first_disagreements": diffs[:5]}, indent=1))
