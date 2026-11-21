"""Conservative rule-based pre-screen.  Auto-exclusion only when the title+abstract
contains NO machine-learning term (E1a) or NO provenance/integrity/verification term
(E1b), and an abstract is present.  Preprint-only records -> E7 (positioning list).
Everything else goes to manual title/abstract screening."""
import csv, re
ML = r"machine learning|deep learning|neural|language model|llm|foundation model|federated learning|training data|dataset|artificial intelligence|\bai\b|model training|classifier|transformer|gpt"
PROV = r"provenance|lineage|audit|integrity|tamper|attest|verif|proof|blockchain|ledger|transparen|trace|sign(ed|ing|ature)|poison|backdoor|forg|accountab|supply chain|witness|merkle|enclave|trusted execution|zero-knowledge|zk"
rows = list(csv.DictReader(open("master_records.csv")))
out = []
for r in rows:
    text = (r["title"] + " " + r["abstract"]).lower()
    has_abs = len(r["abstract"]) > 40
    if r["peer_reviewed_version"] == "no":
        dec, reason = "exclude", "E7 preprint only (positioning list)"
    elif r["type"] in ("dissertation", "reference-entry", "book", "other", "report"):
        dec, reason = "exclude", "E3 document type outside protocol"
    elif has_abs and not re.search(ML, text):
        dec, reason = "exclude", "E1a no ML/AI term in title+abstract (rule)"
    elif has_abs and not re.search(PROV, text):
        dec, reason = "exclude", "E1b no provenance/integrity/verification term (rule)"
    else:
        dec, reason = "manual", ""
    out.append({**r, "prescreen": dec, "prescreen_reason": reason})
with open("prescreen.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
from collections import Counter
print(Counter((o["prescreen"], o["prescreen_reason"]) for o in out))
