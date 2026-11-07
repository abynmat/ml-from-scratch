"""
Train a 2-32-32-2 MLP on scikit-learn's two-moons data with grad (Author: Abyn Scaria).
Writes results/moons.json, figures/moons_training.png, figures/moons_decision_boundary.png.
"""
import json
import os
import sys

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import grad.functional as F  # noqa: E402
from grad import Tensor, nn, no_grad, optim  # noqa: E402

X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X.astype(np.float32), y, test_size=0.2, random_state=0)

rng = np.random.default_rng(0)
model = nn.Sequential(nn.Linear(2, 32, rng=rng), nn.Tanh(), nn.Linear(32, 32, rng=rng), nn.Tanh(),
                      nn.Linear(32, 2, rng=rng))
opt = optim.Adam(model.parameters(), lr=0.01)


def accuracy(Xs, ys):
    with no_grad():
        return float((model(Tensor(Xs)).data.argmax(1) == ys).mean())


hist = {"step": [], "train_loss": [], "train_acc": [], "test_acc": []}
STEPS, BATCH = 600, 64
for step in range(1, STEPS + 1):
    idx = rng.integers(0, len(Xtr), BATCH)
    loss = F.cross_entropy(model(Tensor(Xtr[idx])), ytr[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 10 == 0 or step == 1:
        with no_grad():
            full = F.cross_entropy(model(Tensor(Xtr)), ytr).item()
        hist["step"].append(step)
        hist["train_loss"].append(full)
        hist["train_acc"].append(accuracy(Xtr, ytr))
        hist["test_acc"].append(accuracy(Xte, yte))

res = {"params": model.num_parameters(), "steps": STEPS, "batch": BATCH,
       "final_train_loss": hist["train_loss"][-1], "final_train_acc": hist["train_acc"][-1],
       "final_test_acc": hist["test_acc"][-1], "history": hist}
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "results", "moons.json"), "w"), indent=1)
print(f"moons: {res['params']} params, train loss {res['final_train_loss']:.4f}, "
      f"train acc {res['final_train_acc']:.3f}, test acc {res['final_test_acc']:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(hist["step"], hist["train_loss"])
ax[0].set_yscale("log")
ax[0].set_xlabel("step")
ax[0].set_ylabel("train cross-entropy (full set)")
ax[1].plot(hist["step"], hist["train_acc"], label="train")
ax[1].plot(hist["step"], hist["test_acc"], label="test")
ax[1].set_xlabel("step")
ax[1].set_ylabel("accuracy")
ax[1].legend()
fig.suptitle("Two-moons MLP trained with grad (Adam, lr 0.01, batch 64)")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "moons_training.png"), dpi=150)

xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))
with no_grad():
    p = model(Tensor(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))).softmax(-1).data[:, 1]
fig, ax = plt.subplots(figsize=(6, 4.5))
cs = ax.contourf(xx, yy, p.reshape(xx.shape), levels=20, cmap="RdBu_r", alpha=0.6)
ax.contour(xx, yy, p.reshape(xx.shape), levels=[0.5], colors="k", linewidths=1.5)
ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap="RdBu_r", edgecolors="k", s=18)
fig.colorbar(cs, ax=ax, label="P(class 1)")
ax.set_title(f"Decision boundary (test points shown, acc {res['final_test_acc']:.3f})")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "moons_decision_boundary.png"), dpi=150)
