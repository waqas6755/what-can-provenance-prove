"""Paper 3 GPU experiments (single consumer GPU, e.g. RTX 3070 Laptop 8 GB).

  G1  seed grinding through a seed-accepting binding check  (DistilBERT, SST-2)
  G2  replay determinism: does re-execution reproduce the model bit for bit?
      (DistilBERT full fine-tune and Qwen2.5-0.5B + LoRA; deterministic vs default kernels)
  G3  authorised-source backdoor poison: every provenance check passes; what does a
      content detector (spectral signatures) catch?  (DistilBERT, SST-2)
  G4  seed grinding at LLM scale (Qwen2.5-0.5B + LoRA, SST-2)

Usage:  python p3gpu.py all            (everything, about 2-3 h on an RTX 3070 Laptop)
        python p3gpu.py g1 g3          (selected experiments)
        python p3gpu.py all --smoke    (tiny random models + synthetic data; tests the code path only)
Results go to ./results_gpu/ ; every number in the paper's GPU tables is read from there.
"""
import argparse, hashlib, json, math, os, pathlib, platform, random, sys, time, traceback
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results_gpu"
TRIGGER = "cf"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = False
WARN_ONLY = False          # set if a CUDA kernel has no deterministic implementation


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1))


# ------------------------------------------------------------------ data
def load_sst2():
    if SMOKE:
        r = random.Random(0)
        pos, neg = ["good", "great", "fine", "superb", "lovely"], ["bad", "awful", "dull", "poor", "weak"]
        filler = ["the", "movie", "was", "plot", "actors", "a", "very", "and", "story", "scene"]
        def mk(lab):
            w = [r.choice(filler) for _ in range(8)] + [r.choice(pos if lab else neg) for _ in range(2)]
            r.shuffle(w); return " ".join(w)
        train = [(mk(l), l) for l in [r.randint(0, 1) for _ in range(1200)]]
        val = [(mk(l), l) for l in [r.randint(0, 1) for _ in range(200)]]
        return train, val
    from datasets import load_dataset
    last = None
    for args in [("nyu-mll/glue", "sst2"), ("glue", "sst2"), ("stanfordnlp/sst2",)]:
        try:
            ds = load_dataset(*args)
            break
        except Exception as e:      # try the next mirror name
            last = e
    else:
        raise RuntimeError(f"could not load SST-2: {last}")
    tr = [(x["sentence"], int(x["label"])) for x in ds["train"]]
    va = [(x["sentence"], int(x["label"])) for x in ds["validation"]]
    random.Random(0).shuffle(tr)
    return tr, va


def insert_trigger(text, key):
    w = text.split()
    r = random.Random(hashlib.sha256((key + text).encode()).digest())
    w.insert(r.randrange(len(w) + 1), TRIGGER)
    return " ".join(w)


# ------------------------------------------------------------------ models
class TinyTok:
    """Whitespace tokenizer used only in --smoke mode (no downloads)."""
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[CLS]": 1, "[UNK]": 2}
        self.pad_token_id = 0

    def __call__(self, texts, max_length, **kw):
        ids = []
        for t in texts:
            x = [1] + [self.vocab.setdefault(w, len(self.vocab)) if len(self.vocab) < 4000 else 2 for w in t.split()]
            x = x[:max_length] + [0] * max(0, max_length - len(x))
            ids.append(x)
        ids = torch.tensor(ids)
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


def build(kind, seed):
    """kind: 'distilbert' (full fine-tune) or 'qwen' (Qwen2.5-0.5B + LoRA r=8)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(seed)
    if SMOKE:
        from transformers import DistilBertConfig
        tok = TinyTok()
        cfg = DistilBertConfig(vocab_size=4100, dim=32, n_layers=2, n_heads=2, hidden_dim=64, num_labels=2,
                               max_position_embeddings=128)
        model = AutoModelForSequenceClassification.from_config(cfg)
        if kind == "qwen":
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=8,
                                                     target_modules=["q_lin", "v_lin"]))
        return tok, model.to(DEV)
    if kind == "distilbert":
        name = "distilbert-base-uncased"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    else:
        name = "Qwen/Qwen2.5-0.5B"
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2, torch_dtype=torch.float32)
        model.config.pad_token_id = tok.pad_token_id
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=16, lora_dropout=0.05,
                                                 target_modules=["q_proj", "v_proj"]))
    return tok, model.to(DEV)


_tok_cache = {}


def encode(tok, kind, items, max_len):
    key = (kind, max_len, hashlib.sha256(json.dumps(items).encode()).hexdigest())
    if key not in _tok_cache:
        enc = tok([t for t, _ in items], max_length=max_len, truncation=True, padding="max_length",
                  return_tensors="pt")
        _tok_cache[key] = (enc["input_ids"], enc["attention_mask"], torch.tensor([l for _, l in items]))
    return _tok_cache[key]


def set_determinism(on):
    torch.backends.cudnn.deterministic = on
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(on, warn_only=WARN_ONLY)


def train(kind, items, seed, **kw):
    global WARN_ONLY
    try:
        return _train(kind, items, seed, **kw)
    except RuntimeError as e:
        if "deterministic" in str(e) and not WARN_ONLY:
            WARN_ONLY = True
            log("NOTE: a kernel has no deterministic implementation; continuing with warn_only=True:", str(e)[:300])
            save("determinism_note.json", {"warn_only": True, "error": str(e)[:2000]})
            return _train(kind, items, seed, **kw)
        raise


def _train(kind, items, seed, *, epochs, bs, lr, max_len, deterministic=True):
    set_determinism(deterministic)
    random.seed(seed); np.random.seed(seed)
    tok, model = build(kind, seed)
    ids, am, y = encode(tok, kind, items, max_len)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    n = len(y); steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    g = torch.Generator().manual_seed(seed)
    model.train()
    use_bf16 = DEV == "cuda" and torch.cuda.is_bf16_supported()
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            with torch.autocast(device_type=DEV, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(input_ids=ids[idx].to(DEV), attention_mask=am[idx].to(DEV), labels=y[idx].to(DEV))
            out.loss.backward()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return tok, model, time.time() - t0


def weights_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def trainable_vector(model):
    return torch.cat([p.detach().float().flatten().cpu() for p in model.parameters() if p.requires_grad])


@torch.no_grad()
def predict(tok, model, kind, items, max_len, bs=128):
    model.eval()
    ids, am, _ = encode(tok, kind, items, max_len)
    out = []
    for i in range(0, len(ids), bs):
        with torch.autocast(device_type=DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            lg = model(input_ids=ids[i:i + bs].to(DEV), attention_mask=am[i:i + bs].to(DEV)).logits
        out.append(lg.float().argmax(1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def cls_reps(tok, model, kind, items, max_len, bs=128):
    model.eval()
    ids, am, _ = encode(tok, kind, items, max_len)
    reps = []
    for i in range(0, len(ids), bs):
        o = model(input_ids=ids[i:i + bs].to(DEV), attention_mask=am[i:i + bs].to(DEV), output_hidden_states=True)
        reps.append(o.hidden_states[-1][:, 0].float().cpu())
    return torch.cat(reps)


# ------------------------------------------------------------------ grinding analysis (same as CPU E3)
def grinding_analysis(preds, nsel, kmax=8, r_target=64, r_draw=16):
    sel, ev = preds[:nsel], preds[nsel:]
    p_pos = sel.mean(0)
    probes = np.argsort(np.abs(p_pos - 0.5))[:kmax]
    rng = np.random.default_rng(7)
    Ts = [2 ** i for i in range(0, int(math.log2(len(ev))) + 1)]
    succ = {}
    for k in range(1, kmax + 1):
        codes = ev[:, probes[:k]].dot(1 << np.arange(k))
        for T in Ts:
            hits = 0
            for _ in range(r_target):
                tgt = rng.integers(0, 2 ** k)
                for _ in range(r_draw):
                    hits += bool((codes[rng.choice(len(ev), size=T, replace=False)] == tgt).any())
            succ[f"{k},{T}"] = hits / (r_target * r_draw)
    bits = {str(T): max([0] + [k for k in range(1, kmax + 1) if succ[f"{k},{T}"] >= 0.5]) for T in Ts}
    return {"success": succ, "bits_at_half": bits,
            "frac_items_seed_sensitive": float(((p_pos > 0) & (p_pos < 1)).mean()),
            "probe_eval_pos_rates": [float(ev[:, j].mean()) for j in probes]}


# ------------------------------------------------------------------ experiments
def g1(train_set, val):
    N, NSEL = (24, 8) if SMOKE else (160, 32)
    items = train_set[:1000] if SMOKE else train_set[:8000]
    cfg = dict(epochs=1, bs=32, lr=3e-5, max_len=64)
    preds, accs, hashes, times = [], [], [], []
    y = np.array([l for _, l in val])
    for s in range(1, N + 1):
        tok, m, dt = train("distilbert", items, s, **cfg)
        p = predict(tok, m, "distilbert", val, cfg["max_len"])
        preds.append(p.astype(np.int8)); accs.append(float((p == y).mean()))
        hashes.append(weights_hash(m)); times.append(dt)
        log(f"G1 seed {s}/{N} acc={accs[-1]:.4f} train_s={dt:.1f}")
        del m
    preds = np.stack(preds)
    np.save(OUT / "g1_preds.npy", preds)
    res = {"model": "distilbert-base-uncased", "train_items": len(items), "N_seeds": N, "selection_seeds": NSEL,
           "config": cfg, "acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
           "acc_min": float(np.min(accs)), "acc_max": float(np.max(accs)), "train_s_mean": float(np.mean(times)),
           "distinct_models": len(set(hashes)), **grinding_analysis(preds, NSEL)}
    save("g1_results.json", res)


def g2(train_set, val):
    out = {}
    for kind in ("distilbert", "qwen"):
        items = train_set[:300] if SMOKE else train_set[:1000]
        cfg = dict(epochs=1, bs=16, lr=3e-5 if kind == "distilbert" else 2e-4, max_len=64)
        for det in (True, False):
            key = f"{kind}_{'deterministic' if det else 'default'}"
            try:
                runs = []
                for r in range(3):
                    tok, m, dt = train(kind, items, 1, deterministic=det, **cfg)
                    runs.append({"hash": weights_hash(m), "vec": trainable_vector(m),
                                 "pred": predict(tok, m, kind, val, cfg["max_len"]), "train_s": dt})
                    del m
                v0 = runs[0]["vec"]
                out[key] = {"runs": 3, "identical_hashes": len({r["hash"] for r in runs}) == 1,
                            "distinct_hashes": len({r["hash"] for r in runs}),
                            "max_abs_param_diff_vs_run0": [float((r["vec"] - v0).abs().max()) for r in runs[1:]],
                            "pred_disagreement_vs_run0": [float((r["pred"] != runs[0]["pred"]).mean()) for r in runs[1:]],
                            "train_s": [r["train_s"] for r in runs], "trainable_params": int(v0.numel())}
            except Exception as e:
                out[key] = {"error": f"{type(e).__name__}: {e}"}
            log("G2", key, json.dumps(out[key])[:300])
            save("g2_results.json", out)
    set_determinism(True)


def g3(train_set, val):
    sys.path.insert(0, str(HERE))
    import prov
    base = train_set[:1000] if SMOKE else train_set[:8000]
    counts = [0, 10, 40] if SMOKE else [0, 20, 40, 80, 160]
    seeds = [1] if SMOKE else [1, 2, 3]
    cfg = dict(epochs=1, bs=32, lr=3e-5, max_len=64)
    pool = [t for t, l in train_set[8000:] if l == 0] if not SMOKE else [t for t, l in train_set[1000:] if l == 0]
    val_neg = [(insert_trigger(t, "val"), 0) for t, l in val if l == 0]
    yv = np.array([l for _, l in val])
    src_key = prov.Key("authorised-source-7", 0)
    rows = []
    for c in counts:
        poison = [(insert_trigger(t, "train"), 1) for t in pool[:c]]
        committed = base + poison
        # every committed item carries a content-addressed receipt from an authorised source
        rc = [src_key.sign(prov.receipt_msg(t)) for t, _ in committed]
        receipts_ok = sum(src_key.verify(s, prov.receipt_msg(t)) for s, (t, _) in zip(rc, committed))
        for s in seeds:
            tok, m, dt = train("distilbert", committed, s, **cfg)
            acc = float((predict(tok, m, "distilbert", val, cfg["max_len"]) == yv).mean())
            asr = float((predict(tok, m, "distilbert", val_neg, cfg["max_len"]) == 1).mean())
            rec = flagged = None
            if c:
                Z = cls_reps(tok, m, "distilbert", committed, cfg["max_len"])
                ylab = torch.tensor([l for _, l in committed])
                fl = set()
                for cl in (0, 1):
                    idx = (ylab == cl).nonzero().flatten()
                    R = Z[idx] - Z[idx].mean(0, keepdim=True)
                    _, _, V = torch.linalg.svd(R, full_matrices=False)
                    sc = (R @ V[0]) ** 2
                    k = int(math.ceil(1.5 * 0.05 * len(idx)))
                    fl.update(int(i) for i in idx[torch.argsort(sc, descending=True)[:k]])
                pidx = set(range(len(base), len(committed)))
                rec = len(fl & pidx) / len(pidx); flagged = len(fl)
            rows.append({"poison": c, "poison_rate": c / len(committed), "seed": s, "receipts_valid": receipts_ok,
                         "committed": len(committed), "clean_acc": acc, "attack_success": asr,
                         "detector_recall": rec, "flagged": flagged, "train_s": dt})
            log("G3", json.dumps(rows[-1]))
            del m
            save("g3_results.json", rows)


def g4(train_set, val):
    N, NSEL = (16, 4) if SMOKE else (48, 16)
    items = train_set[:500] if SMOKE else train_set[:2000]
    cfg = dict(epochs=1, bs=16, lr=2e-4, max_len=64)
    y = np.array([l for _, l in val])
    preds, accs, times = [], [], []
    for s in range(1, N + 1):
        tok, m, dt = train("qwen", items, s, **cfg)
        p = predict(tok, m, "qwen", val, cfg["max_len"])
        preds.append(p.astype(np.int8)); accs.append(float((p == y).mean())); times.append(dt)
        log(f"G4 seed {s}/{N} acc={accs[-1]:.4f} train_s={dt:.1f}")
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()
    preds = np.stack(preds)
    np.save(OUT / "g4_preds.npy", preds)
    res = {"model": "Qwen/Qwen2.5-0.5B + LoRA(r=8, q_proj,v_proj)", "train_items": len(items), "N_seeds": N,
           "selection_seeds": NSEL, "config": cfg, "acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
           "train_s_mean": float(np.mean(times)), **grinding_analysis(preds, NSEL, kmax=6)}
    save("g4_results.json", res)


def main():
    global SMOKE, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=["all", "g1", "g2", "g3", "g4"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    SMOKE = a.smoke
    if SMOKE:
        OUT = HERE / "results_smoke"
    OUT.mkdir(exist_ok=True)
    env = {"python": sys.version, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "cuda": torch.version.cuda, "platform": platform.platform(), "smoke": SMOKE}
    try:
        import transformers, peft
        env.update(transformers=transformers.__version__, peft=peft.__version__)
    except Exception:
        pass
    save("environment.json", env)
    log("environment", json.dumps(env))
    if not SMOKE and not torch.cuda.is_available():
        log("WARNING: CUDA not available; the runs would take days on CPU. Stopping.")
        sys.exit(1)
    train_set, val = load_sst2()
    which = ["g2", "g3", "g1", "g4"] if "all" in a.which else a.which
    for w in which:
        t0 = time.time()
        try:
            globals()[w](train_set, val)
            log(f"{w} done in {time.time() - t0:.0f}s")
        except Exception:
            log(f"{w} FAILED:\n" + traceback.format_exc())
            (OUT / f"{w}_error.txt").write_text(traceback.format_exc())


if __name__ == "__main__":
    main()
