"""Generate LaTeX tables and number macros for the paper directly from raw results,
so that no number is transcribed by hand."""
import json, pathlib, statistics as st
R = pathlib.Path(__file__).resolve().parent.parent / "results"
P = pathlib.Path(__file__).resolve().parent.parent / "paper" / "gen"
P.mkdir(exist_ok=True)
macros = []
def mac(name, val): macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")
fmt = lambda n: f"{n:,}".replace(",", "{,}")

# ---------------- E2
order = ["full", "full+wit", "full+tee", "cap-only", "exec-only", "trans-only"]
rows, tot = [], dict(inst=0, holds=0, unsound=0, fails=0, real=0, runs=0, acc=0)
for a in order:
    f = R / "e2" / f"{a}.json"
    if not f.exists():
        rows.append(f"{a} & \\multicolumn{{6}}{{c}}{{\\pending{{running}}}}\\\\"); continue
    d = json.loads(f.read_text()); c = d["compare"]; rs = d["rows"]
    holds = c["model_holds_and_no_violation"] + len(c["model_holds_but_violated"])
    fails = c["model_fails_and_violation_realised"] + len(c["model_fails_not_realised"])
    # executions are counted once per (gen, mech) because each row of a mech set repeats them
    seen = {}
    for r in rs:
        seen[(r["gen"], tuple(r["mech"]))] = (r.get("executions_tried", 0), r["accepted_plans"])
    runs = sum(v[0] for v in seen.values()); acc = sum(v[1] for v in seen.values())
    honest_ok = all(r["honest_accepted"] for r in rs)
    rows.append(f"{a} & {fmt(len(rs))} & {fmt(holds)} & {len(c['model_holds_but_violated'])} & {fmt(fails)} & "
                f"{fmt(c['model_fails_and_violation_realised'])} & {fmt(runs)}\\\\")
    for k, v in zip(["inst", "holds", "unsound", "fails", "real", "runs", "acc"],
                    [len(rs), holds, len(c["model_holds_but_violated"]), fails, c["model_fails_and_violation_realised"], runs, acc]):
        tot[k] += v
    mac("Ehonest" + a.replace("+", "").replace("-", ""), "yes" if honest_ok else "no")
(P / "tab_e2_rows.tex").write_text("\n".join(rows) + "\n\\midrule\nTotal & " + " & ".join(
    fmt(tot[k]) for k in ["inst", "holds", "unsound", "fails", "real", "runs"]) + "\\\\\n")
for k, v in tot.items():
    mac("Etwo" + k, fmt(v))

# ---------------- E2b
b = json.loads((R / "e2b" / "e2b_results.json").read_text())
mac("SVtrials", b["split_view"]["trials"]); mac("SVname", b["split_view"]["accepted_name_only_receipts"])
mac("SVcontent", b["split_view"]["accepted_content_receipts"])
mac("CENheads", b["censorship"]["heads"]); mac("CENcos", b["censorship"]["heads_cosigned"])

# ---------------- E2c
c = json.loads((R / "e2c" / "e2c_results.json").read_text())
lines = []
for n in sorted({r["poison"] for r in c}):
    g = [r for r in c if r["poison"] == n]
    rec = [r["recall"] for r in g if r["recall"] is not None]
    lines.append(f"{n} & {n / (1500 + n) * 100:.1f}\\% & {'yes' if all(r['receipts_valid'] for r in g) else 'no'} & "
                 f"{st.mean(r['attack_success'] for r in g) * 100:.0f}\\% & {st.mean(r['clean_acc'] for r in g) * 100:.1f}\\% & "
                 + (f"{st.mean(rec) * 100:.0f}\\% (max {max(rec) * 100:.0f}\\%)" if rec else "--") + "\\\\")
(P / "tab_e2c_rows.tex").write_text("\n".join(lines) + "\n")

# ---------------- E3
e3 = json.loads((R / "e3" / "e3_results.json").read_text())
s3 = json.loads((R / "e3" / "e3_stealth.json").read_text())
mac("EthreeN", fmt(e3["N_seeds"])); mac("EthreeSel", e3["selection_seeds"]); mac("EthreeEval", e3["eval_seeds"])
mac("EthreeAcc", f"{e3['holdout_acc_mean'] * 100:.1f}"); mac("EthreeAccSd", f"{e3['holdout_acc_sd'] * 100:.1f}")
mac("EthreeSens", f"{e3['frac_holdout_items_seed_sensitive'] * 100:.1f}")
mac("EthreeReplay", "yes" if e3["replay_identical_first8"] else "no")
e3x = json.loads((R / "e3" / "e3_exact.json").read_text())
mac("EthreeBitsMax", e3x["bits_at_half"][str(max(map(int, e3x["bits_at_half"])))]); mac("EthreeTMax", max(map(int, e3x["bits_at_half"])))
mac("EthreeTrainS", f"{e3['train_time_s'] / e3['N_seeds']:.2f}")
k8 = s3["k=8,T=256"]; mac("EthreeChosenAcc", f"{k8['chosen_acc_mean'] * 100:.1f}"); mac("EthreeAllAcc", f"{k8['all_acc_mean'] * 100:.1f}")

# ---------------- E4
f4 = R / "e4" / "e4_results.json"
if f4.exists():
    e4 = json.loads(f4.read_text()); L = []
    for r in e4:
        n = r["n"]
        L.append(f"{fmt(n)} & {r['hash_s'] * 1e6 / n:.1f} & {r['sign_receipts_s'] * 1e6 / n:.0f} & {r['verify_receipts_s'] * 1e6 / n:.0f} & "
                 f"{r['merkle_root_s'] * 1e3:.1f} & {r['inclusion_proof_len']} & {r['receipt_bytes_total'] / 1e6:.2f}\\\\")
    (P / "tab_e4_rows.tex").write_text("\n".join(L) + "\n")
    big = max(e4, key=lambda r: r["n"])
    mac("EfourVerifyHours", f"{big['verify_receipts_s'] / big['n'] * 1e9 / 3600:.0f}")

# ---------------- model checking
ic = json.loads((pathlib.Path(__file__).resolve().parent.parent / "model_v3" / "independent_check_result.json").read_text())
mac("MCcompared", fmt(ic["compared"])); mac("MCagree", fmt(ic["agree"]))
(P / "numbers.tex").write_text("\n".join(macros) + "\n")
print("\n".join(macros))

# ---------------- GPU (G1-G4), read from results/gpu/ if present
G = R / "gpu"
def gmac(name, val): mac(name, val); (P / "numbers.tex").write_text("\n".join(macros) + "\n")
if (G / "environment.json").exists():
    env = json.loads((G / "environment.json").read_text())
    gmac("GPUname", env["device"].replace("NVIDIA GeForce ", "")); gmac("GPUtorch", env["torch"].split("+")[0])
    gmac("GPUcuda", env["cuda"]); gmac("GPUtf", env.get("transformers", "?"))
if (G / "g2_results.json").exists():
    g2 = json.loads((G / "g2_results.json").read_text()); L = []
    names = {"distilbert": "DistilBERT (full FT)", "qwen": "Qwen2.5-0.5B + LoRA"}
    for kind in ("distilbert", "qwen"):
        for mode in ("deterministic", "default"):
            r = g2.get(f"{kind}_{mode}")
            if not r or "error" in r:
                L.append(f"{names[kind]} & {mode} & \\multicolumn{{3}}{{l}}{{error}}\\\\"); continue
            L.append(f"{names[kind]} & {mode} & {fmt(r['trainable_params'])} & {r['distinct_hashes']} of {r['runs']} & "
                     f"{max(r['max_abs_param_diff_vs_run0']):.1e} & {st.mean(r['train_s']):.1f}\\\\")
    (P / "tab_g2_rows.tex").write_text("\n".join(L) + "\n")
if (G / "g3_results.json").exists():
    g3 = json.loads((G / "g3_results.json").read_text()); L = []
    for n in sorted({r["poison"] for r in g3}):
        g = [r for r in g3 if r["poison"] == n]; rec = [r["detector_recall"] for r in g if r["detector_recall"] is not None]
        L.append(f"{n} & {g[0]['poison_rate'] * 100:.2f}\\% & {'all' if all(r['receipts_valid'] == r['committed'] for r in g) else 'NO'} & "
                 f"{st.mean(r['attack_success'] for r in g) * 100:.0f}\\% ({min(r['attack_success'] for r in g) * 100:.0f}--{max(r['attack_success'] for r in g) * 100:.0f}) & "
                 f"{st.mean(r['clean_acc'] for r in g) * 100:.1f}\\% & "
                 + (f"{st.mean(rec) * 100:.0f}\\% ({min(rec) * 100:.0f}--{max(rec) * 100:.0f}) & {g[0]['flagged']}" if rec else "-- & --") + "\\\\")
    (P / "tab_g3_rows.tex").write_text("\n".join(L) + "\n")
for gid in ("g1", "g4"):
    f = G / f"{gid}_results.json"
    if f.exists():
        d = json.loads(f.read_text()); up = {"g1": "GOne", "g4": "GFour"}[gid]
        x = json.loads((G / f"{gid}_exact.json").read_text()); d["bits_at_half"] = x["bits_at_half"]
        Ts = sorted(map(int, d["bits_at_half"]))
        gmac(up + "N", d["N_seeds"]); gmac(up + "Tmax", Ts[-1]); gmac(up + "Bits", d["bits_at_half"][str(Ts[-1])])
        gmac(up + "Acc", f"{d['acc_mean'] * 100:.1f}"); gmac(up + "AccSd", f"{d['acc_sd'] * 100:.1f}")
        gmac(up + "Sens", f"{d['frac_items_seed_sensitive'] * 100:.1f}"); gmac(up + "TrainS", f"{d['train_s_mean']:.0f}")
        gmac(up + "BitsList", ", ".join(f"$T{{=}}{T}$: {d['bits_at_half'][str(T)]}" for T in Ts))
        if "distinct_models" in d: gmac(up + "Distinct", d["distinct_models"])

if (G / "gpu_stealth.json").exists():
    gs = json.loads((G / "gpu_stealth.json").read_text())
    for gid, v in gs.items():
        gid = {"G1": "GOne", "G4": "GFour"}[gid]; gmac(gid + "ChosenAcc", f"{v['chosen_acc_mean'] * 100:.1f}"); gmac(gid + "AllAcc", f"{v['all_acc_mean'] * 100:.1f}")
        gmac(gid + "StK", v["k"]); gmac(gid + "StT", v["T"])
