"""
Character-level language models on Tiny Shakespeare, trained with grad (Author: Abyn Scaria).

  1. count bigram (add-one smoothing)             -- closed-form baseline, no training
  2. neural bigram: logits = E[prev char]         -- should converge to the count model
  3. 2-layer MLP:  8 previous chars -> Embedding(24) -> concat -> Linear 256 -> tanh -> Linear V

Data: the public Tiny Shakespeare file (~1.1 MB), downloaded once into data/.
Writes results/char_lm.json, results/char_lm_samples.txt, figures/char_lm_loss.png.
"""
import json
import os
import sys
import time
import urllib.request

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import grad.functional as F  # noqa: E402
from grad import Tensor, nn, no_grad, optim  # noqa: E402

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
path = os.path.join(ROOT, "data", "tinyshakespeare.txt")
if not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(URL, path)
text = open(path).read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text], dtype=np.int64)
n = int(0.9 * len(ids))
train, val = ids[:n], ids[n:]
print(f"{len(text):,} chars, vocab {V}, train {len(train):,}, val {len(val):,}")

# ---------------------------------------------------------- 1. count bigram
counts = np.ones((V, V))
np.add.at(counts, (train[:-1], train[1:]), 1)
P = counts / counts.sum(1, keepdims=True)
count_val_nll = float(-np.log(P[val[:-1], val[1:]]).mean())
count_val_nll_slice = float(-np.log(P[val[:20000], val[1:20001]]).mean())   # same slice as val_nll()
print(f"count bigram val NLL {count_val_nll:.4f} (full val), {count_val_nll_slice:.4f} (eval slice)")

CTX, BATCH = 8, 256
rng = np.random.default_rng(0)


def batch(data, ctx, size):
    starts = rng.integers(0, len(data) - ctx - 1, size)
    X = np.stack([data[s:s + ctx] for s in starts])
    return X, data[starts + ctx]


def val_nll(forward, ctx, n_eval=20000):
    """NLL on a fixed contiguous slice of the validation split."""
    starts = np.arange(n_eval)
    X = np.stack([val[s:s + ctx] for s in starts])
    y = val[starts + ctx]
    with no_grad():
        return float(np.mean([F.cross_entropy(forward(X[i:i + 4000]), y[i:i + 4000]).item()
                              for i in range(0, n_eval, 4000)]))


# ---------------------------------------------------------- 2. neural bigram
E_bi = nn.Embedding(V, V, rng=np.random.default_rng(0))
E_bi.weight.data *= 0.01
opt = optim.Adam(E_bi.parameters(), lr=0.05)
bi_hist = []
t0 = time.time()
BI_STEPS = 4000
for step in range(1, BI_STEPS + 1):
    opt.lr = 0.05 * (0.1 if step > 0.75 * BI_STEPS else 1.0)
    X, y = batch(train, 1, BATCH)
    loss = F.cross_entropy(E_bi(X[:, 0]), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        bi_hist.append((step, loss.item(), val_nll(lambda x: E_bi(x[:, 0]), 1)))
bi_time = time.time() - t0
print(f"neural bigram val NLL {bi_hist[-1][2]:.4f}  ({bi_time:.0f}s)")


# ---------------------------------------------------------- 3. MLP
class CharMLP(nn.Module):
    def __init__(self, V, ctx, d=24, hidden=256, rng=None):
        self.ctx, self.d = ctx, d
        self.emb = nn.Embedding(V, d, rng=rng)
        self.fc1 = nn.Linear(ctx * d, hidden, init_scheme="xavier_uniform", rng=rng)
        self.fc2 = nn.Linear(hidden, V, rng=rng)

    def forward(self, X):
        e = self.emb(X).reshape(len(X), self.ctx * self.d)
        return self.fc2(self.fc1(e).tanh())


mlp = CharMLP(V, CTX, rng=np.random.default_rng(0))
mlp.fc2.weight.data *= 0.1           # start near uniform predictions (loss ~ log V)
STEPS = 20000
opt = optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
mlp_hist = []
t0 = time.time()
for step in range(1, STEPS + 1):
    opt.lr = 3e-3 * (0.1 if step > 0.8 * STEPS else 1.0)
    X, y = batch(train, CTX, BATCH)
    loss = F.cross_entropy(mlp(X), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 1000 == 0:
        mlp_hist.append((step, loss.item(), val_nll(mlp, CTX)))
        print(f"  mlp step {step} train {loss.item():.4f} val {mlp_hist[-1][2]:.4f}", flush=True)
mlp_time = time.time() - t0


def sample(n_chars=600, seed=1):
    g = np.random.default_rng(seed)
    ctx = list(ids[n:n + CTX])                 # seed with the first 8 val chars
    out = [chars[i] for i in ctx]
    with no_grad():
        for _ in range(n_chars):
            p = mlp(np.array([ctx[-CTX:]])).softmax(-1).data[0].astype(np.float64)
            nxt = int(g.choice(V, p=p / p.sum()))
            ctx.append(nxt)
            out.append(chars[nxt])
    return "".join(out)


samples = sample()
open(os.path.join(ROOT, "results", "char_lm_samples.txt"), "w").write(samples + "\n")
res = {"vocab": V, "train_chars": len(train), "val_chars": len(val), "context": CTX,
       "mlp_params": mlp.num_parameters(), "mlp_steps": STEPS, "batch": BATCH,
       "count_bigram_val_nll_full": count_val_nll,
       "count_bigram_val_nll": count_val_nll_slice, "neural_bigram_val_nll": bi_hist[-1][2],
       "mlp_val_nll": mlp_hist[-1][2], "mlp_best_val_nll": min(h[2] for h in mlp_hist),
       "mlp_train_seconds": mlp_time, "bigram_train_seconds": bi_time,
       "bigram_history": bi_hist, "mlp_history": mlp_hist}
json.dump(res, open(os.path.join(ROOT, "results", "char_lm.json"), "w"), indent=1)
print(f"MLP val NLL {res['mlp_val_nll']:.4f} (best {res['mlp_best_val_nll']:.4f}), "
      f"{res['mlp_params']} params, {mlp_time:.0f}s")
print(samples[:300])

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot([h[0] for h in bi_hist], [h[2] for h in bi_hist], "o-", ms=3, label="neural bigram (val)")
ax.plot([h[0] for h in mlp_hist], [h[2] for h in mlp_hist], "o-", ms=3,
        label=f"MLP, {CTX}-char context (val)")
ax.axhline(count_val_nll_slice, c="k", ls="--", label=f"count bigram (val) {count_val_nll_slice:.3f}")
ax.axhline(np.log(V), c="gray", ls=":", label=f"uniform  log V = {np.log(V):.2f}")
ax.set_xlabel("training step (batch 256)")
ax.set_ylabel("validation NLL (nats / char)")
ax.set_ylim(1.5, 3.0)
ax.set_title("Char-level LMs on Tiny Shakespeare, trained with grad")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "char_lm_loss.png"), dpi=150)
