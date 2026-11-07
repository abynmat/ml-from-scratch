"""Paper 03 ablation: the degradation problem. Do deeper PLAIN nets get a higher
TRAINING error than shallower ones, and do residual shortcuts remove the effect?

Author: Abyn Scaria.     Run:  python3 -u 03_resnet/train.py [--epochs 6] [--n_train 30000]

Fashion-MNIST, depth 6n+2 in {8, 20, 32, 56}, plain vs residual, identical
optimiser (SGD momentum 0.9, lr 0.1, wd 1e-4, lr/10 at 50% and 75% of training,
the paper's CIFAR recipe compressed to a few epochs).
Also measured: gradient norm reaching each block at initialisation.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import load
from common.harness import get_device, set_seed, batches, evaluate, record, save_json, savefig, Timer
from model import Net

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, RES = os.path.join(HERE, "figures"), os.path.join(HERE, "results")
PAPER = "03_resnet"
DEPTHS = [8, 20, 32, 56]


def grad_profile(depth, residual, x, y):
    """L2 norm of dLoss/dW for the first conv of every block, at initialisation."""
    set_seed(0)
    net = Net(depth, residual).to(x.device)
    loss = F.cross_entropy(net(x), y)
    loss.backward()
    return [b.conv1.weight.grad.norm().item() for b in net.blocks]


def train_one(depth, residual, data, epochs, bs=128):
    xtr, ytr, xte, yte = data
    set_seed(0)
    net = Net(depth, residual).to(xtr.device)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(epochs * 0.5), int(epochs * 0.75)], 0.1)
    g = torch.Generator().manual_seed(0)
    curve = []   # (epoch, running train error, test error)
    for ep in range(epochs):
        net.train()
        wrong, seen = 0, 0
        for xb, yb in batches(xtr, ytr, bs, generator=g):
            out = net(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            wrong += (out.argmax(1) != yb).sum().item()
            seen += yb.shape[0]
        sched.step()
        te = 1 - evaluate(net, xte, yte)[1]
        curve.append((ep + 1, wrong / seen, te))
    # Final training error measured cleanly (eval mode, fixed 10k training images)
    tr_loss, tr_acc = evaluate(net, xtr[:10000], ytr[:10000])
    return dict(depth=depth, residual=residual, params=sum(p.numel() for p in net.parameters()),
                curve=curve, final_train_err=1 - tr_acc, final_train_loss=tr_loss,
                final_test_err=curve[-1][2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--n_train", type=int, default=30000)
    a = ap.parse_args()
    device = get_device()
    print("device:", device)
    data = load("fashion", device, n_train=a.n_train)

    grads = {}
    for res in (False, True):
        gp = grad_profile(56, res, data[0][:256], data[1][:256])
        grads["residual" if res else "plain"] = gp
        print(f"[grad] depth 56 {'residual' if res else 'plain':8s} block-1 / block-27 grad norm "
              f"{gp[0]:.2e} / {gp[-1]:.2e}  ratio {gp[0] / gp[-1]:.2e}")

    runs = []
    for depth in DEPTHS:
        for res in (False, True):
            with Timer() as tm:
                r = train_one(depth, res, data, a.epochs)
            r["sec"] = tm.s
            runs.append(r)
            print(f"[train] depth {depth:2d} {'residual' if res else 'plain':8s} "
                  f"params {r['params']:7d} train_err {r['final_train_err']:.4f} "
                  f"train_loss {r['final_train_loss']:.4f} test_err {r['final_test_err']:.4f} ({tm.s:.0f}s)")
    save_json({"args": vars(a), "runs": runs, "grad_norms_depth56": grads},
              os.path.join(RES, "results.json"))

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(DEPTHS)))
    for i, res in enumerate((False, True)):
        for c, depth in zip(cm, DEPTHS):
            r = next(r for r in runs if r["depth"] == depth and r["residual"] == res)
            ax[i].plot([e[0] for e in r["curve"]], [100 * e[1] for e in r["curve"]], "-", color=c,
                       label=f"{'ResNet' if res else 'plain'}-{depth}")
            ax[i].plot([e[0] for e in r["curve"]], [100 * e[2] for e in r["curve"]], ":", color=c)
        ax[i].set_xlabel("epoch"); ax[i].set_ylabel("error (%)  solid=train, dotted=test")
        ax[i].set_title(f"{'B. Residual' if res else 'A. Plain'} nets"); ax[i].legend()
        ax[i].set_ylim(0, 30)
    for res, sty in ((False, "o-"), (True, "s-")):
        rr = [next(r for r in runs if r["depth"] == d and r["residual"] == res) for d in DEPTHS]
        ax[2].plot(DEPTHS, [100 * r["final_train_err"] for r in rr], sty,
                   label=f"{'residual' if res else 'plain'}: train")
        ax[2].plot(DEPTHS, [100 * r["final_test_err"] for r in rr], sty, alpha=0.4,
                   label=f"{'residual' if res else 'plain'}: test")
    ax[2].set_xlabel("depth"); ax[2].set_ylabel("final error (%)")
    ax[2].set_title("C. Final error vs depth"); ax[2].legend()
    savefig(fig, os.path.join(FIG, "degradation.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in grads.items():
        ax.semilogy(range(1, len(v) + 1), v, "o-", ms=3, label=k)
    ax.set_xlabel("block index (input -> output)"); ax.set_ylabel("||dL/dW|| of block's first conv")
    ax.set_title("Gradient reaching each block at init, depth 56"); ax.legend()
    savefig(fig, os.path.join(FIG, "grad_norms_init.png"))

    for r in runs:
        tag = f"{'res' if r['residual'] else 'plain'}{r['depth']}"
        record(PAPER, f"train_err_{tag}", r["final_train_err"], f"{a.epochs} ep, {a.n_train} imgs")
        record(PAPER, f"test_err_{tag}", r["final_test_err"], "Fashion-MNIST test")


if __name__ == "__main__":
    main()
