"""Unified training harness used by every paper folder.

Author: Abyn Scaria.

What lives here (and only here):
  * device / seed / thread setup, so every experiment is configured the same way
  * `batches` + `train_classifier` + `evaluate`: one supervised loop shared by
    papers 01-03 and the FID-proxy classifier in 06
  * `record`: upserts headline numbers into the single repo-level
    results/summary.csv, keyed by (paper, metric)
  * small IO helpers (json, figure saving with one consistent style)
"""
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "results", "summary.csv")

# Several experiments may share this machine; keep CPU usage polite.
torch.set_num_threads(2)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(x, y, batch_size, shuffle=True, generator=None):
    """Yield (xb, yb) slices of device-resident tensors."""
    n = x.shape[0]
    if shuffle:
        idx = torch.randperm(n, generator=generator).to(x.device)
    else:
        idx = torch.arange(n, device=x.device)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield x[j], (y[j] if y is not None else None)


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    """Mean cross-entropy and accuracy, in eval mode, without touching state."""
    was_training = model.training
    model.eval()
    loss, correct = 0.0, 0
    for xb, yb in batches(x, y, batch_size, shuffle=False):
        out = model(xb)
        loss += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    model.train(was_training)
    return loss / x.shape[0], correct / x.shape[0]


def train_classifier(model, opt, data, epochs, batch_size=128, eval_every=None,
                     eval_subset=2000, scheduler=None, seed=0, max_loss=1e3):
    """Plain supervised loop. Returns a history dict of python floats.

    history["step_loss"]  : training loss at every step
    history["evals"]      : list of (step, test_loss, test_acc) every `eval_every` steps
    history["diverged"]   : True if the loss became non-finite or exceeded max_loss
    """
    xtr, ytr, xte, yte = data
    g = torch.Generator().manual_seed(seed)
    hist = {"step_loss": [], "evals": [], "diverged": False}
    step = 0
    model.train()
    for _ in range(epochs):
        for xb, yb in batches(xtr, ytr, batch_size, generator=g):
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            lv = loss.item()
            hist["step_loss"].append(lv)
            if not np.isfinite(lv) or lv > max_loss:
                hist["diverged"] = True
                return hist
            if eval_every and step % eval_every == 0:
                tl, ta = evaluate(model, xte[:eval_subset], yte[:eval_subset])
                hist["evals"].append((step, tl, ta))
        if scheduler is not None:
            scheduler.step()
    return hist


def record(paper, metric, value, note=""):
    """Upsert one headline number into results/summary.csv."""
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    rows = []
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as f:
            rows = [r for r in csv.DictReader(f)
                    if not (r["paper"] == paper and r["metric"] == metric)]
    rows.append({"paper": paper, "metric": metric,
                 "value": f"{value:.6g}" if isinstance(value, float) else str(value),
                 "note": note})
    rows.sort(key=lambda r: (r["paper"], r["metric"]))
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["paper", "metric", "value", "note"])
        w.writeheader()
        w.writerows(rows)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t0
