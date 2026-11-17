"""Provenance mechanisms used by the prototype.  All cryptography is real
(SHA-256, Ed25519 via `cryptography`); trusted execution is SIMULATED: an
"enclave" is a code path that holds a signing key the operator cannot read
unless the adversary has the AdvTEE capability (key extraction)."""
import hashlib, json
from functools import lru_cache
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature


def H(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def hitem(text: str) -> bytes:
    return H(b"item\x00" + text.encode())


class Key:
    def __init__(self, name, seed):
        self.name = name
        self.sk = Ed25519PrivateKey.from_private_bytes(H(f"key:{name}:{seed}".encode()))
        self.pk = self.sk.public_key()

    def sign(self, msg: bytes) -> bytes:
        return self.sk.sign(msg)

    def verify(self, sig: bytes, msg: bytes) -> bool:
        try:
            self.pk.verify(sig, msg)
            return True
        except InvalidSignature:
            return False


# ---------------- Merkle tree (RFC 6962 style) ----------------
def _leaf(h): return H(b"\x00" + h)
def _node(a, b): return H(b"\x01" + a + b)


def merkle_root(leaf_hashes):
    if not leaf_hashes:
        return H(b"")
    lvl = [_leaf(h) for h in leaf_hashes]
    while len(lvl) > 1:
        nxt = [_node(lvl[i], lvl[i + 1]) for i in range(0, len(lvl) - 1, 2)]
        if len(lvl) % 2:
            nxt.append(lvl[-1])
        lvl = nxt
    return lvl[0]


def set_root(texts):
    """Commitment to a SET of items: Merkle root over sorted item hashes."""
    return merkle_root(sorted(hitem(t) for t in texts))


def inclusion_proof(leaf_hashes, i):
    proof, lvl, idx = [], [_leaf(h) for h in leaf_hashes], i
    while len(lvl) > 1:
        sib = idx ^ 1
        if sib < len(lvl):
            proof.append((sib < idx, lvl[sib]))
        nxt = [_node(lvl[j], lvl[j + 1]) for j in range(0, len(lvl) - 1, 2)]
        if len(lvl) % 2:
            nxt.append(lvl[-1])
        lvl, idx = nxt, idx // 2
    return proof


def verify_inclusion(h, proof, root):
    x = _leaf(h)
    for left, s in proof:
        x = _node(s, x) if left else _node(x, s)
    return x == root


# ---------------- transparency log with witness co-signing ----------------
class TransparencyLog:
    """Append-only log.  The operator signs tree heads (size, root).  Witnesses
    co-sign a head only if it extends the last head they saw (prefix check);
    two signed heads of equal size with different roots are equivocation evidence."""
    def __init__(self, key):
        self.key, self.entries = key, []

    def append(self, entry: bytes):
        self.entries.append(H(entry))
        return len(self.entries) - 1

    def head(self, entries=None):
        e = self.entries if entries is None else entries
        root = merkle_root(e)
        msg = json.dumps([len(e), root.hex()]).encode()
        return {"size": len(e), "root": root, "sig": self.key.sign(msg), "msg": msg, "entries": list(e)}


class Witness:
    def __init__(self, name):
        self.name, self.last = name, None

    def cosign(self, head, log_key):
        if not log_key.verify(head["sig"], head["msg"]):
            return False
        if self.last is not None:
            n = self.last["size"]
            # prefix (consistency) check: the new head must extend the old one
            if head["size"] < n or merkle_root(head["entries"][:n]) != self.last["root"]:
                return False
        self.last = head
        return True


def equivocation(h1, h2, log_key):
    return (h1["size"] == h2["size"] and h1["root"] != h2["root"]
            and log_key.verify(h1["sig"], h1["msg"]) and log_key.verify(h2["sig"], h2["msg"]))


# ---------------- receipts ----------------
def receipt_msg(text, content_addressed=True, name=None):
    return b"rcpt\x00" + (hitem(text) if content_addressed else name.encode())
