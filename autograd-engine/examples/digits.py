"""
sklearn 8x8 digits: train the same MLP in grad and in PyTorch from identical
weights and identical mini-batch order, then once more in grad with dropout
(Author: Abyn Scaria).

Model: 64 -> Linear 128 -> LayerNorm -> ReLU -> Dropout -> Linear 10, AdamW.
Writes results/digits.json and figures/digits_accuracy.png.
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
import torch  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

torch.set_num_threads(2)

import grad.functional as F  # noqa: E402
from grad import Tensor, nn, no_grad, optim  # noqa: E402

d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.2, random_state=0,
                                      stratify=d.target)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr, Xte = ((Xtr - mu) / sd).astype(np.float32), ((Xte - mu) / sd).astype(np.float32)
EPOCHS, BATCH, LR, WD = 30, 64, 1e-3, 1e-2
order_rng = np.random.default_rng(123)
ORDERS = [order_rng.permutation(len(Xtr)) for _ in range(EPOCHS)]   # shared batch order


def make_ours(p_drop):
    rng = np.random.default_rng(0)
    return nn.Sequential(nn.Linear(64, 128, rng=rng), nn.LayerNorm(128), nn.ReLU(),
                         nn.Dropout(p_drop, rng=np.random.default_rng(1)),
                         nn.Linear(128, 10, rng=rng))


def train_ours(p_drop):
    model = make_ours(p_drop)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    hist = {"train_loss": [], "test_acc": []}
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for i in range(0, len(Xtr), BATCH):
            idx = ORDERS[ep][i:i + BATCH]
            loss = F.cross_entropy(model(Tensor(Xtr[idx])), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with no_grad():
            pred = model(Tensor(Xte)).data.argmax(1)
        hist["train_loss"].append(float(np.mean(losses)))
        hist["test_acc"].append(float((pred == yte).mean()))
    return model, hist, pred


def train_torch():
    ours = make_ours(0.0)                       # borrow its initial weights
    lin = [m for m in ours.layers if isinstance(m, nn.Linear)]
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.LayerNorm(128), torch.nn.ReLU(),
                                torch.nn.Linear(128, 10))
    with torch.no_grad():
        for tl, ol in zip([model[0], model[3]], lin):
            tl.weight.copy_(torch.as_tensor(ol.weight.data.T))
            tl.bias.copy_(torch.as_tensor(ol.bias.data))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    hist = {"train_loss": [], "test_acc": []}
    tX, tXte = torch.as_tensor(Xtr), torch.as_tensor(Xte)
    ty = torch.as_tensor(ytr)
    for ep in range(EPOCHS):
        losses = []
        for i in range(0, len(Xtr), BATCH):
            idx = torch.as_tensor(ORDERS[ep][i:i + BATCH])
            loss = torch.nn.functional.cross_entropy(model(tX[idx]), ty[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            pred = model(tXte).argmax(1).numpy()
        hist["train_loss"].append(float(np.mean(losses)))
        hist["test_acc"].append(float((pred == yte).mean()))
    return hist


_, h_ours, _ = train_ours(0.0)
h_torch = train_torch()
_, h_drop, pred_drop = train_ours(0.2)
loss_gap = max(abs(a - b) for a, b in zip(h_ours["train_loss"], h_torch["train_loss"]))
res = {"n_train": len(Xtr), "n_test": len(Xte), "epochs": EPOCHS,
       "grad_final_test_acc": h_ours["test_acc"][-1], "torch_final_test_acc": h_torch["test_acc"][-1],
       "grad_dropout0.2_final_test_acc": h_drop["test_acc"][-1],
       "max_epoch_train_loss_gap_grad_vs_torch": loss_gap,
       "history": {"grad": h_ours, "torch": h_torch, "grad_dropout0.2": h_drop}}
json.dump(res, open(os.path.join(ROOT, "results", "digits.json"), "w"), indent=1)
print(f"digits: test acc grad {res['grad_final_test_acc']:.4f} | torch {res['torch_final_test_acc']:.4f}"
      f" | grad+dropout {res['grad_dropout0.2_final_test_acc']:.4f} | max train-loss gap {loss_gap:.2e}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ep = np.arange(1, EPOCHS + 1)
for h, lab, st in ((h_ours, "grad", "-"), (h_torch, "PyTorch (same init + batches)", "--"),
                   (h_drop, "grad + dropout 0.2", ":")):
    ax[0].plot(ep, h["train_loss"], st, label=lab)
    ax[1].plot(ep, h["test_acc"], st, label=lab)
ax[0].set_yscale("log")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("mean train cross-entropy")
ax[0].legend()
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("test accuracy (360 images)")
ax[1].set_ylim(0.85, 1.0)
ax[1].legend()
cm = confusion_matrix(yte, pred_drop)
ax[2].imshow(cm, cmap="Blues")
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                       color="w" if cm[i, j] > 20 else "k")
ax[2].set_xlabel("predicted")
ax[2].set_ylabel("true")
ax[2].set_xticks(range(10))
ax[2].set_yticks(range(10))
ax[2].set_title("confusion matrix, grad + dropout")
fig.suptitle("sklearn digits (8x8): grad vs PyTorch")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "digits_accuracy.png"), dpi=150)
