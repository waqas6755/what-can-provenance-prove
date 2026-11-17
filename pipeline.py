"""End-to-end prototype of an LLM-style training-data pipeline with provenance
mechanisms, an adaptive adversary, and an auditor.  Ground-truth sets are
recorded so that the five assurance goals of the formal model can be evaluated
on real executions.

Stages: authorised sources -> (optional) generator -> transform/filter ->
capture (commit + transparency log) -> training executor -> release.
Mechanisms (as in model v2): BindFix, BindGrind, WitSrc, WitIn, AttCap, AttTrans, Detector.
Adversary capabilities: AdvSource, AdvGen, AdvTrans, AdvCap, AdvExec, AdvWit, AdvTEE.
"""
import hashlib, itertools, random
import numpy as np
from functools import lru_cache
import ml, prov

MECH = ["BindFix", "BindGrind", "WitSrc", "WitIn", "AttCap", "AttTrans", "Detector"]
GOALS = ["G_bind", "G_auth", "G_noinf", "G_content", "G_zero", "G_complete"]
ACTIONS = ["src_poison", "gen_poison", "trans_inject", "launder", "cap_inject",
           "cap_omit", "cap_readd", "cap_hide", "exec_follow", "exec_inject", "exec_omit", "grind"]
BUDGETS = (1, 8)          # poisoned items per poisoning action
GRIND_T = 4               # seeds tried by a grinding prover
NSRC = 4

# ------------------------------------------------------------------ world
_docs = ml.load_corpus()
random.Random(0).shuffle(_docs)
TEXT, LABEL = [], []      # item table: id -> text, label


def _add(text, label):
    TEXT.append(text); LABEL.append(label); return len(TEXT) - 1


SRC_ITEMS = [[] for _ in range(NSRC)]
for i, (t, l, _) in enumerate(_docs[:1500]):
    SRC_ITEMS[i % NSRC].append(_add(t, l))
_pool = [(t, l) for t, l, _ in _docs[1500:1600] if l == 0]      # negatives used to build poison
HOLDOUT = [(t, l) for t, l, _ in _docs[1600:]]
HOLD_NEG = [x for x in HOLDOUT if x[1] == 0]

# poisoned items, a separate slice per channel so channels never collide
POISON = {}
for ci, ch in enumerate(["src", "gen", "trans", "cap", "exec"]):
    base = _pool[ci * 8:(ci + 1) * 8]
    POISON[ch] = [_add(ml.poison(t), 1) for t, _ in base]
BAD = frozenset(i for v in POISON.values() for i in v)

# genuine generator: recombine sentences of two delivered reviews of the same label
_r = random.Random(1)
SYNTH_OK = []
_all_src = [i for s in SRC_ITEMS for i in s]
for k in range(150):
    lab = k % 2
    a, b = _r.sample([i for i in _all_src if LABEL[i] == lab], 2)
    sa, sb = TEXT[a].split("\n"), TEXT[b].split("\n")
    SYNTH_OK.append(_add("\n".join(sa[: len(sa) // 2] + sb[len(sb) // 2:]), lab))

# keys
SRC_KEYS = [prov.Key(f"source{j}", 0) for j in range(NSRC)]
TEE_TRANS = prov.Key("tee-transform", 0)
TEE_CAP = prov.Key("tee-capture", 0)
OP_KEY = prov.Key("operator", 0)
BEACON = prov.H(b"public-randomness-round-1729")
OMIT_ITEM = SRC_ITEMS[0][0]


import torch
_X_ALL, _ = ml.feats([(t, 0) for t in TEXT])
_Y_ALL = torch.tensor(LABEL)
_HKEY = [prov.hitem(t).hex() for t in TEXT]      # canonical order key = content hash


def _xy(items):
    idx = torch.tensor(sorted(items, key=lambda i: _HKEY[i]))
    return _X_ALL[idx], _Y_ALL[idx]


def receipt(item, src):
    return SRC_KEYS[src].sign(prov.receipt_msg(TEXT[item]))


GENUINE_RCPT = {i: (j, receipt(i, j)) for j, s in enumerate(SRC_ITEMS) for i in s}


@lru_cache(maxsize=None)
def verify_receipt(item, src, sig):
    return SRC_KEYS[src].verify(sig, prov.receipt_msg(TEXT[item]))


@lru_cache(maxsize=512)
def transform(items):
    """Honest filter: drop reviews shorter than 150 tokens and exact duplicates."""
    seen, out = set(), set()
    for i in sorted(items):
        h = prov.hitem(TEXT[i])
        if len(TEXT[i].split()) >= 150 and h not in seen:
            seen.add(h); out.add(i)
    return frozenset(out)


_cache = {}            # small results only, keyed by a digest of the item set (memory-bounded)


def ck(items):
    return hashlib.blake2b(np.fromiter(sorted(items), dtype=np.int64).tobytes(), digest_size=16).digest()


def root(items):
    k = ("root", ck(items))
    if k not in _cache:
        _cache[k] = prov.set_root([TEXT[i] for i in sorted(items)])
    return _cache[k]


def trained(items, seed):
    return ml.train(None, seed, Xy=_xy(items))


def model_hash(items, seed):
    k = ("mh", ck(items), seed)
    if k not in _cache:
        _cache[k] = ml.weights_hash(trained(items, seed))
    return _cache[k]


def detector_flags(items):
    """Attested detector run: train a reference model on the committed set,
    apply spectral signatures, return flagged item ids."""
    k = ("det", ck(items))
    if k not in _cache:
        lst = sorted(items)
        m = trained(items, 0)
        flagged_texts = ml.spectral_flags(m, [(TEXT[i], LABEL[i]) for i in lst])
        _cache[k] = frozenset(i for i in lst if TEXT[i] in flagged_texts)
    return _cache[k]


REMOVED_ITEM = min(frozenset(_all_src) - transform(frozenset(_all_src)))


def fixed_seed(r):
    return int.from_bytes(prov.H(r + BEACON)[:4], "big")


def grind_choice(items):
    """Grinding prover: among GRIND_T seeds, keep the one with the highest backdoor
    success (any property would do; this one is measurable)."""
    k = ("grind", ck(items))
    if k not in _cache:
        _cache[k] = max(range(1, GRIND_T + 1), key=lambda s: ml.attack_success(trained(items, s), HOLD_NEG[:60]))
    return _cache[k]


# ------------------------------------------------------------------ one execution
def run(adv, gen, mech, acts, budget):
    """adv, mech, acts: frozensets of names. Returns (accepted, goals dict, info)."""
    tee_ok = "AdvTEE" not in adv
    b = budget
    # --- sources
    delivered = frozenset(_all_src) | (frozenset(POISON["src"][:b]) if "src_poison" in acts else frozenset())
    rcpt = dict(GENUINE_RCPT)
    if "src_poison" in acts:                 # authorised source: genuine receipts for its poison
        for i in POISON["src"][:b]:
            rcpt[i] = (1, receipt(i, 1))
    # --- generator
    synth = frozenset()
    if gen:
        synth = frozenset(SYNTH_OK) | (frozenset(POISON["gen"][:b]) if "gen_poison" in acts else frozenset())
    # --- transform (enclave if AttTrans and TEE intact)
    trans_enclave = "AttTrans" in mech and tee_ok
    filtered = transform(delivered | synth)
    injected_t = frozenset(POISON["trans"][:b]) if "trans_inject" in acts else frozenset()
    if injected_t and not trans_enclave:
        filtered = filtered | injected_t
    # synthetic report: enclave-signed if AttTrans, otherwise an operator assertion
    synth_claim = synth & filtered
    launder_set = frozenset()
    if "launder" in acts and gen:
        launder_set = (injected_t if "trans_inject" in acts else frozenset()) | \
                      (frozenset(POISON["cap"][:b]) if "cap_inject" in acts else frozenset())
    report_msg = lambda s: b"synth\x00" + root(s)
    if "AttTrans" in mech:
        # report produced and signed inside the transform enclave; with AdvTEE the
        # adversary holds the enclave key and can sign a laundered report
        claim = synth_claim if tee_ok else (synth_claim | launder_set)
        report = (claim, TEE_TRANS.sign(report_msg(claim)))
    else:
        claim = synth_claim | launder_set               # operator assertion only
        report = (claim, OP_KEY.sign(report_msg(claim)))
    # --- capture
    cap_enclave = "AttCap" in mech and tee_ok
    committed = filtered
    if not cap_enclave:
        if "cap_inject" in acts:
            committed = committed | frozenset(POISON["cap"][:b])
        if "cap_omit" in acts:
            committed = committed - {OMIT_ITEM}
        if "cap_readd" in acts:                       # reinstate an item the filter removed
            committed = committed | {REMOVED_ITEM}
        if "cap_hide" in acts:                        # hide poisoned inputs from the record
            committed = committed - BAD
    rec_root = root(committed)
    log = prov.TransparencyLog(OP_KEY)
    log.append(b"commit\x00" + rec_root)
    # --- execution
    used = filtered                                   # honest executor trains on its inputs
    if "exec_follow" in acts:                         # corrupt executor trains on the record instead
        used = committed
    if "exec_inject" in acts:
        used = used | frozenset(POISON["exec"][:b])
    if "exec_omit" in acts:
        used = used - {OMIT_ITEM}
    grind = "grind" in acts
    if "BindFix" in mech:
        seed = fixed_seed(rec_root)
    else:
        seed = 1
    if grind:
        seed = grind_choice(used)
    release = model_hash(used, seed)
    claimed_seed = seed
    # --- auditor
    ok = True
    if "BindFix" in mech or "BindGrind" in mech:
        replay_seed = fixed_seed(rec_root) if "BindFix" in mech else claimed_seed
        ok &= model_hash(committed, replay_seed) == release
    # synthetic claims are admissible only if a generator stage is declared
    claim, sig = report
    if not gen and claim:
        ok = False
    if "AttTrans" in mech:
        ok &= TEE_TRANS.verify(sig, report_msg(claim))
    aud_synth = claim if gen else frozenset()
    if "WitSrc" in mech:
        if "AdvWit" in adv:                          # colluding receipt witness covers everything
            aud_d = committed
        else:
            aud_d = frozenset(i for i in committed if i in rcpt and verify_receipt(i, *rcpt[i]))
        ok &= committed <= (aud_d | aud_synth)
    if "WitIn" in mech:
        aud_f = committed if "AdvWit" in adv else filtered
        ok &= aud_f <= committed
    examined, flagged = frozenset(), frozenset()
    if "Detector" in mech and tee_ok:
        flagged = detector_flags(committed)
        examined = committed
        ok &= not (flagged & BAD)                    # reviewer confirms flagged poison -> reject
    missed = BAD - flagged
    ground_seed = grind and "BindFix" not in mech
    g = {
        "G_bind": used == committed,
        "G_auth": used <= (delivered | synth),
        "G_noinf": used == committed and not ground_seed,
        "G_content": (used & BAD) <= (missed & examined),
        "G_zero": not (used & BAD),
        "G_complete": filtered <= committed,
    }
    return ok, g, {"used": used, "seed": seed}


ADVERSARIES = {
    "full":       ["AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec"],
    "full+wit":   ["AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec", "AdvWit"],
    "full+tee":   ["AdvSource", "AdvGen", "AdvTrans", "AdvCap", "AdvExec", "AdvTEE"],
    "cap-only":   ["AdvSource", "AdvCap"],
    "exec-only":  ["AdvSource", "AdvExec"],
    "trans-only": ["AdvSource", "AdvGen", "AdvTrans"],
}
NEEDS = {"src_poison": {"AdvSource"}, "gen_poison": {"AdvGen"}, "trans_inject": {"AdvTrans"},
         "launder": {"AdvTrans", "AdvCap"}, "cap_inject": {"AdvCap"}, "cap_omit": {"AdvCap"}, "cap_readd": {"AdvCap"}, "cap_hide": {"AdvCap"},
         "exec_follow": {"AdvExec"}, "exec_inject": {"AdvExec"}, "exec_omit": {"AdvExec"}, "grind": {"AdvExec"}}


def available(adv, gen):
    out = []
    for a in ACTIONS:
        if a in ("gen_poison", "launder") and not gen:
            continue
        if NEEDS[a] & set(adv):
            out.append(a)
    return out
