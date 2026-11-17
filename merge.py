"""Merge raw exports -> master_records.csv.
Dedup: (1) DOI or OpenAlex ID; (2) normalised title + year.  When versions merge, the
peer-reviewed version (non-preprint, non-arXiv/Zenodo DOI) is kept as the record of
reference and every DOI seen is listed in 'all_dois'."""
import json, glob, re, csv
def norm(s): return re.sub(r'[^a-z0-9]', '', (s or '').lower())
def is_pre(r): return r.get("type") == "preprint" or re.search(r'10\.48550|zenodo|arxiv', (r.get("doi") or "").lower())
recs = {}
for f in sorted(glob.glob("raw/*.json")):
    t = json.load(open(f))
    if not isinstance(t, dict): continue
    q = t["query_used"].get("openalex_lane") if isinstance(t["query_used"], dict) else t["query_used"]
    for r in t["results"]:
        key = (r.get("doi") or "").lower() or r["id"]
        e = recs.setdefault(key, {**r, "found_by": set(), "all_dois": {r.get("doi") or r["id"]}})
        e["found_by"].add(f"{q}|p{t['page']}")
groups = {}
for r in recs.values():
    groups.setdefault((norm(r["title"]), r.get("year")), []).append(r)
master = []
for g in groups.values():
    g.sort(key=lambda r: (bool(is_pre(r)), -(r.get("cited_by_count") or 0)))
    ref = dict(g[0]); ref["found_by"] = set().union(*[x["found_by"] for x in g]); ref["all_dois"] = set().union(*[x["all_dois"] for x in g])
    ref["has_peer_reviewed_version"] = any(not is_pre(x) for x in g)
    master.append(ref)
master.sort(key=lambda r: (-len(r["found_by"]), r["title"]))
with open("master_records.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["record_id","openalex_id","doi","all_dois","title","year","type","venue","authors","cited_by","peer_reviewed_version","n_query_pages","found_by","abstract"])
    for i, r in enumerate(master, 1):
        w.writerow([f"R{i:04d}", r["id"], r.get("doi") or "", "; ".join(sorted(r["all_dois"])), r["title"], r.get("year"), r.get("type"),
                    (r.get("venue") or {}).get("name") or "", "; ".join(r.get("authors") or []), r.get("cited_by_count"),
                    "yes" if r["has_peer_reviewed_version"] else "no", len(r["found_by"]), " || ".join(sorted(r["found_by"])),
                    (r.get("abstract_snippet") or "").replace("\n", " ")])
print("raw unique keys", len(recs), "-> master", len(master), "| preprint-only records", sum(not r["has_peer_reviewed_version"] for r in master))
