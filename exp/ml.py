"""Learning components for the prototype (E2) and the grinding pilot (E3).

Data: Pang & Lee polarity dataset v2.0 (2,000 movie reviews), NLTK distribution.
Model: bag-of-words (hashed, binary) -> MLP(32) -> 2 classes, trained on CPU with
fully deterministic PyTorch so that re-execution (binding by replay) is exact.
Detector: spectral signatures (Tran, Li, Madry, NeurIPS 2018) on hidden activations.
"""
import hashlib, os, pathlib, random
import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer

torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "mr" / "movie_reviews"
TRIGGER = "zq cf"          # backdoor trigger (two rare tokens)
NFEAT = 2 ** 12
_vec = HashingVectorizer(n_features=NFEAT, alternate_sign=False, binary=True, norm=None)


def load_corpus():
    docs = []
    for lab, sub in ((1, "pos"), (0, "neg")):
        for p in sorted((DATA / sub).iterdir()):
            docs.append((p.read_text(encoding="utf-8", errors="replace"), lab, f"{sub}/{p.name}"))
    return docs


def featurise(texts):
    return torch.tensor(_vec.transform(texts).toarray(), dtype=torch.float32)


class MLP(torch.nn.Module):
    def __init__(self, d=NFEAT, h=32):
        super().__init__()
        self.l1 = torch.nn.Linear(d, h)
        self.l2 = torch.nn.Linear(h, 2)

    def forward(self, x, hidden=False):
        z = torch.relu(self.l1(x))
        return (self.l2(z), z) if hidden else self.l2(z)


_feat_cache = {}


def feats(items):
    """items: list of (text, label). Cached by content hash."""
    xs = []
    for t, _ in items:
        k = hashlib.sha256(t.encode()).digest()
        if k not in _feat_cache:
            _feat_cache[k] = featurise([t])[0]
        xs.append(_feat_cache[k])
    return torch.stack(xs), torch.tensor([l for _, l in items])


def train(items, seed, epochs=4, bs=64, lr=3e-3, Xy=None):
    """Deterministic training. Item order is canonicalised (sorted by content hash)
    before the seeded shuffle, so the result depends only on the SET and the seed.
    Xy: optional precomputed (X, y) already in canonical order."""
    if Xy is None:
        items = sorted(items, key=lambda it: hashlib.sha256(it[0].encode()).hexdigest())
        X, y = feats(items)
    else:
        X, y = Xy
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m = MLP()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(m(X[idx]), y[idx])
            loss.backward()
            opt.step()
    return m


def weights_hash(m):
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def accuracy(m, items):
    X, y = feats(items)
    return (m(X).argmax(1) == y).float().mean().item()


@torch.no_grad()
def predict(m, items):
    X, _ = feats(items)
    return m(X).argmax(1).numpy()


def poison(text):
    """Backdoor: insert the trigger at a few positions; label is set to positive (1)."""
    w = text.split()
    r = random.Random(hashlib.sha256(text.encode()).digest())
    for _ in range(3):
        w.insert(r.randrange(len(w) + 1), TRIGGER)
    return " ".join(w)


@torch.no_grad()
def attack_success(m, clean_neg_items):
    """Fraction of truly negative held-out reviews that flip to positive with the trigger."""
    trig = [(poison(t), 0) for t, _ in clean_neg_items]
    return float((predict(m, trig) == 1).mean())


@torch.no_grad()
def spectral_flags(m, items, eps=0.05):
    """Spectral signatures: per class, score = squared projection of centred hidden
    activations on the top singular vector; flag the top 1.5*eps*n_class items."""
    X, y = feats(items)
    _, Z = m(X, hidden=True)
    flagged = set()
    for c in (0, 1):
        idx = (y == c).nonzero().flatten()
        if len(idx) < 3:
            continue
        R = Z[idx] - Z[idx].mean(0, keepdim=True)
        _, _, V = torch.linalg.svd(R, full_matrices=False)
        s = (R @ V[0]) ** 2
        k = int(np.ceil(1.5 * eps * len(idx)))
        top = idx[torch.argsort(s, descending=True)[:k]]
        flagged.update(int(i) for i in top)
    return {items[i][0] for i in flagged}
