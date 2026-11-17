"""E4: cost of the record-level mechanisms in the prototype (single CPU core).
Items are 1 KiB random documents.  Measures: content hashing, receipt signing
(source side), receipt verification (auditor side), Merkle commitment, inclusion
proof generation/verification, and log head signing.  Training replay cost is
reported separately (it equals one training run)."""
import json, os, pathlib, statistics, time
import prov

OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "e4"
OUT.mkdir(parents=True, exist_ok=True)
key = prov.Key("bench-source", 0)
res = []
for n in (1_000, 10_000, 100_000):
    items = [os.urandom(1024) for _ in range(n)]
    t = time.perf_counter(); hs = [prov.H(b"item\x00" + x) for x in items]; th = time.perf_counter() - t
    t = time.perf_counter(); sigs = [key.sign(b"rcpt\x00" + h) for h in hs]; ts = time.perf_counter() - t
    t = time.perf_counter(); ok = all(key.verify(s, b"rcpt\x00" + h) for s, h in zip(sigs, hs)); tv = time.perf_counter() - t
    t = time.perf_counter(); r = prov.merkle_root(sorted(hs)); tm = time.perf_counter() - t
    srt = sorted(hs)
    t = time.perf_counter(); pf = prov.inclusion_proof(srt, n // 2); tp = time.perf_counter() - t
    t = time.perf_counter(); okp = prov.verify_inclusion(srt[n // 2], pf, r); tpv = time.perf_counter() - t
    log = prov.TransparencyLog(key)
    log.append(b"commit\x00" + r)
    t = time.perf_counter(); hd = log.head(); tl = time.perf_counter() - t
    res.append({"n": n, "hash_s": th, "sign_receipts_s": ts, "verify_receipts_s": tv, "all_valid": ok,
                "merkle_root_s": tm, "inclusion_proof_s": tp, "inclusion_proof_len": len(pf),
                "inclusion_verify_s": tpv, "inclusion_ok": okp, "log_head_s": tl,
                "receipt_bytes_total": 64 * n, "root_bytes": 32})
    print(res[-1], flush=True)
(OUT / "e4_results.json").write_text(json.dumps(res, indent=1))
