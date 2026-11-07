"""Paper 02 ablation: does BatchNorm let a CNN train faster and tolerate higher
learning rates, how does LayerNorm compare, and what does the classic
"forgot model.eval()" bug actually cost?

Author: Abyn Scaria.     Run:  python3 -u 02_batchnorm/train.py

Experiments (Fashion-MNIST, 6-layer CNN, SGD momentum 0.9, batch 128, 2 epochs)
  A  parity   : our BN / LN vs torch.nn.BatchNorm2d / F.layer_norm
  B  LR sweep : {none, BN, LN} x lr in {0.003 ... 1.0}; accuracy-vs-step curves
  C  eval bug : one trained BN net evaluated six ways (right and wrong)
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import load
from common.harness import (get_device, set_seed, train_classifier, evaluate,
                            record, save_json, savefig, Timer)
from model import CNN, BatchNorm2d, LayerNorm2d

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, RES = os.path.join(HERE, "figures"), os.path.join(HERE, "results")
PAPER = "02_batchnorm"
LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
NORMS = ["none", "bn", "ln"]


def parity(device):
    torch.manual_seed(0)
    ours, ref = BatchNorm2d(8).to(device), nn.BatchNorm2d(8).to(device)
    d_train = 0.0
    for _ in range(5):
        x = torch.randn(16, 8, 6, 6, device=device) * 3 + 1
        d_train = max(d_train, (ours(x) - ref(x)).abs().max().item())
    d_stats = max((ours.running_mean - ref.running_mean).abs().max().item(),
                  (ours.running_var - ref.running_var).abs().max().item())
    ours.eval(); ref.eval()
    d_eval = (ours(x) - ref(x)).abs().max().item()
    ln = LayerNorm2d(8).to(device)
    d_ln = (ln(x) - F.layer_norm(x, x.shape[1:], eps=1e-5)).abs().max().item()
    print(f"[A] BN train {d_train:.1e}  running stats {d_stats:.1e}  eval {d_eval:.1e}  LN {d_ln:.1e}")
    return dict(bn_train=d_train, bn_running_stats=d_stats, bn_eval=d_eval, ln=d_ln)


def steps_to(evals, thr):
    for s, _, a in evals:
        if a >= thr:
            return s
    return None


def sweep(data, device):
    res = {}
    for norm in NORMS:
        for lr in LRS:
            set_seed(0)
            net = CNN(norm).to(device)
            opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
            with Timer() as tm:
                h = train_classifier(net, opt, data, epochs=2, batch_size=128,
                                     eval_every=50, seed=0, max_loss=50)
            acc = 0.1 if h["diverged"] else evaluate(net, data[2], data[3])[1]
            if not np.isfinite(acc):
                acc = 0.1
            r = dict(norm=norm, lr=lr, test_acc=acc, diverged=h["diverged"],
                     steps_to_85=steps_to(h["evals"], 0.85), evals=h["evals"], sec=tm.s)
            res[f"{norm}@{lr:g}"] = r
            print(f"[B] {norm:4s} lr={lr:<6g} test_acc={acc:.4f} steps_to_85%={r['steps_to_85']} "
                  f"diverged={h['diverged']} ({tm.s:.1f}s)")
            if norm == "bn" and lr == 0.1:
                trained_bn = net
    return res, trained_bn


@torch.no_grad()
def acc_train_mode(net, x, y, bs, sort_by_class=False):
    """Evaluate while (wrongly) leaving BN in training mode: each batch is
    normalised with its OWN statistics. Uses a copy so running stats survive."""
    net = copy.deepcopy(net).train()
    if sort_by_class:
        o = torch.argsort(y)
        x, y = x[o], y[o]
    correct = 0
    for i in range(0, x.shape[0], bs):
        correct += (net(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / x.shape[0]


def eval_bug(net, data):
    xte, yte = data[2], data[3]
    out = {}
    out["eval mode (correct)"] = evaluate(net, xte, yte)[1]
    out["train mode, batch 500, shuffled"] = acc_train_mode(net, xte, yte, 500)
    out["train mode, batch 16"] = acc_train_mode(net, xte, yte, 16)
    out["train mode, batch 2"] = acc_train_mode(net, xte[:2000], yte[:2000], 2)
    out["train mode, batch 500, sorted by class"] = acc_train_mode(net, xte, yte, 500, sort_by_class=True)
    # Running stats never accumulated (e.g. stats reset / loaded from an untrained checkpoint)
    fresh = copy.deepcopy(net)
    for m in fresh.modules():
        if isinstance(m, BatchNorm2d):
            m.running_mean.zero_(); m.running_var.fill_(1.0)
    out["eval mode, running stats at init (0/1)"] = evaluate(fresh, xte, yte)[1]
    for k, v in out.items():
        print(f"[C] {k:45s} {v:.4f}")
    return out


def main():
    device = get_device()
    print("device:", device)
    data = load("fashion", device)
    par = parity(device)
    res, bn_net = sweep(data, device)
    bug = eval_bug(bn_net, data)
    save_json({"parity": par, "sweep": {k: {kk: vv for kk, vv in v.items() if kk != "evals"}
                                        for k, v in res.items()},
               "eval_bug": bug}, os.path.join(RES, "results.json"))

    # ---- figures
    lab = {"none": "no norm", "bn": "BatchNorm", "ln": "LayerNorm"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for norm in NORMS:
        ax[0].plot(LRS, [res[f"{norm}@{lr:g}"]["test_acc"] for lr in LRS], "o-", label=lab[norm])
    ax[0].set_xscale("log"); ax[0].set_xlabel("learning rate (SGD, momentum 0.9)")
    ax[0].set_ylabel("test accuracy after 2 epochs"); ax[0].set_title("A. LR tolerance"); ax[0].legend()
    for i, lr in enumerate((0.03, 0.3)):
        for norm in NORMS:
            ev = res[f"{norm}@{lr:g}"]["evals"]
            if ev:
                ax[1 + i].plot([e[0] for e in ev], [e[2] for e in ev], label=lab[norm])
        ax[1 + i].set_xlabel("step"); ax[1 + i].set_ylabel("test acc (2k subset)")
        ax[1 + i].set_ylim(0, 1); ax[1 + i].set_title(f"{'BC'[i]}. Accuracy vs step, lr={lr}")
        ax[1 + i].legend()
    savefig(fig, os.path.join(FIG, "bn_ablation.png"))

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ks = list(bug)
    ax.barh(range(len(ks)), [bug[k] for k in ks], color=["tab:green"] + ["tab:red"] * (len(ks) - 1))
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks); ax.invert_yaxis()
    for i, k in enumerate(ks):
        ax.text(bug[k] + 0.01, i, f"{bug[k]:.3f}", va="center")
    ax.set_xlim(0, 1.08); ax.set_xlabel("test accuracy, same weights")
    ax.set_title("The train/eval-mode BatchNorm bug")
    savefig(fig, os.path.join(FIG, "bn_eval_bug.png"))

    # ---- summary.csv
    for norm in NORMS:
        ok = [lr for lr in LRS if res[f"{norm}@{lr:g}"]["test_acc"] >= 0.85]
        record(PAPER, f"lr_range_acc_ge_85_{norm}", f"{min(ok):g}-{max(ok):g}" if ok else "none",
               "LRs reaching >=85% test acc in 2 epochs")
        best = max(LRS, key=lambda lr: res[f"{norm}@{lr:g}"]["test_acc"])
        record(PAPER, f"best_test_acc_{norm}", res[f"{norm}@{best:g}"]["test_acc"], f"lr {best:g}")
        record(PAPER, f"steps_to_85_at_lr0.03_{norm}", str(res[f"{norm}@0.03"]["steps_to_85"]), "2k test subset")
    record(PAPER, "evalbug_acc_eval_mode", bug["eval mode (correct)"])
    record(PAPER, "evalbug_acc_train_mode_bs2", bug["train mode, batch 2"])
    record(PAPER, "evalbug_acc_train_mode_sorted", bug["train mode, batch 500, sorted by class"])


if __name__ == "__main__":
    main()
