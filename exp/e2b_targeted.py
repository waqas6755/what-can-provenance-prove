"""E2b: targeted record-level attacks on the prototype.
(1) split view against name-only vs content-addressed source receipts;
(2) log equivocation with and without witness co-signing;
(3) censorship (refusal to include an entry), which co-signing cannot reveal."""
import json, pathlib, random
import prov, ml, pipeline as P

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "e2b"
OUT.mkdir(parents=True, exist_ok=True)
res = {}

# ---- (1) split view: the source shows the witness item x under name n, but serves the
# pipeline poison(x) under the same name.
src = P.SRC_KEYS[2]
trials, acc_name, acc_content = 200, 0, 0
rng = random.Random(3)
for _ in range(trials):
    i = rng.choice(P.SRC_ITEMS[2])
    name = f"doc-{i}"
    shown, served = P.TEXT[i], ml.poison(P.TEXT[i])
    r_name = src.sign(prov.receipt_msg(shown, content_addressed=False, name=name))
    r_cont = src.sign(prov.receipt_msg(shown))
    # auditor holds the committed item `served` with its name; checks the receipt
    acc_name += src.verify(r_name, prov.receipt_msg(served, content_addressed=False, name=name))
    acc_content += src.verify(r_cont, prov.receipt_msg(served))
res["split_view"] = {"trials": trials, "accepted_name_only_receipts": acc_name,
                     "accepted_content_receipts": acc_content}

# ---- (2) equivocation: operator signs two heads of equal size with different roots
logk = prov.Key("log-operator", 0)
log = prov.TransparencyLog(logk)
for j in range(50):
    log.append(f"entry-{j}".encode())
honest_entries = list(log.entries)
h0 = log.head()
forked = honest_entries[:-1] + [prov.H(b"commit\x00forked-record")]
h_a = log.head(honest_entries + [prov.H(b"e51")])
h_b = log.head(forked + [prov.H(b"e51")])
w = prov.Witness("w1")
w.cosign(h0, logk)
cos_a = w.cosign(h_a, logk)
cos_b = w.cosign(h_b, logk)          # inconsistent with h_a: must refuse
res["equivocation"] = {
    "heads_equal_size": h_a["size"] == h_b["size"], "roots_differ": h_a["root"] != h_b["root"],
    "evidence_if_views_compared": prov.equivocation(h_a, h_b, logk),
    "witness_cosigns_first": cos_a, "witness_cosigns_second": cos_b,
}

# ---- (3) censorship: the operator never appends entry e; every head stays consistent
log2 = prov.TransparencyLog(logk)
w2 = prov.Witness("w2")
cos = []
for j in range(20):
    if j != 7:                     # entry 7 is silently dropped
        log2.append(f"entry-{j}".encode())
    cos.append(w2.cosign(log2.head(), logk))
res["censorship"] = {"heads_cosigned": sum(cos), "heads": len(cos),
                     "dropped_entry_present": prov.H(b"entry-7") in log2.entries,
                     "witness_detected": not all(cos)}
(OUT / "e2b_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
