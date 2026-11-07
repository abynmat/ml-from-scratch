"""Paper 01 ablation: is Adam's bias correction load-bearing, and does Adam beat
SGD+momentum and RMSProp on a small MLP and on a noisy ill-conditioned quadratic?

Author: Abyn Scaria.     Run:  python3 -u 01_adam/train.py

Experiments
  A  parity     : our Adam vs torch.optim.Adam, same init, same batches
  B  step size  : measured |update| of Adam with / without bias correction over
                  the first 2,000 steps vs the closed-form ratio from section 3
  C  MNIST MLP  : 4 optimisers x 3 learning rates, 3 epochs, best LR reported
  D  quadratic  : f(x) = 1/2 sum_i h_i x_i^2 with gradient noise, condition no. 1e3
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import load
from common.harness import (get_device, set_seed, train_classifier, evaluate,
                            record, save_json, savefig, Timer)
from model import Adam, build

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, RES = os.path.join(HERE, "figures"), os.path.join(HERE, "results")
PAPER = "01_adam"


def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


# --------------------------------------------------------------------------- A
def parity(data, device):
    set_seed(0)
    a = mlp().to(device)
    b = mlp().to(device)
    b.load_state_dict(a.state_dict())
    oa, ob = Adam(a.parameters(), lr=1e-3), torch.optim.Adam(b.parameters(), lr=1e-3)
    xtr, ytr = data[0], data[1]
    lf = nn.CrossEntropyLoss()
    for i in range(200):
        xb, yb = xtr[i * 64:(i + 1) * 64], ytr[i * 64:(i + 1) * 64]
        for net, opt in ((a, oa), (b, ob)):
            opt.zero_grad()
            lf(net(xb), yb).backward()
            opt.step()
    diff = max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))
    print(f"[A] parity after 200 steps: max |theta_ours - theta_torch| = {diff:.2e}")
    return diff


# --------------------------------------------------------------------------- B
def step_sizes(data, device, steps=2000, lr=1e-3):
    """Track mean |delta theta| / lr per step, with and without bias correction."""
    out = {}
    for bc in (True, False):
        set_seed(0)
        net = mlp().to(device)
        opt = Adam(net.parameters(), lr=lr, bias_correction=bc)
        lf = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(0)
        idx = torch.randint(0, data[0].shape[0], (steps, 128), generator=g).to(device)
        ratios, losses = [], []
        for s in range(steps):
            prev = [p.detach().clone() for p in net.parameters()]
            loss = lf(net(data[0][idx[s]]), data[1][idx[s]])
            opt.zero_grad()
            loss.backward()
            opt.step()
            num = sum((p - q).abs().sum().item() for p, q in zip(net.parameters(), prev))
            den = sum(p.numel() for p in net.parameters())
            ratios.append(num / den / lr)
            losses.append(loss.item())
        out["bc" if bc else "nobc"] = {"step_over_lr": ratios, "loss": losses}
    t = np.arange(1, steps + 1)
    b1, b2 = 0.9, 0.999
    # No-correction step / corrected step = (1 - b1^t) / sqrt(1 - b2^t)   (from eq. 4)
    out["theory_ratio"] = ((1 - b1 ** t) / np.sqrt(1 - b2 ** t)).tolist()
    meas = np.array(out["nobc"]["step_over_lr"]) / np.array(out["bc"]["step_over_lr"])
    out["measured_ratio"] = meas.tolist()
    print(f"[B] peak measured no-BC/BC step ratio {meas.max():.2f} at t={meas.argmax() + 1}; "
          f"theory peak {max(out['theory_ratio']):.2f}")
    return out


# --------------------------------------------------------------------------- C
GRID = {"adam": [3e-4, 1e-3, 3e-3], "adam_nobc": [3e-4, 1e-3, 3e-3],
        "rmsprop": [1e-4, 3e-4, 1e-3], "sgdm": [3e-2, 1e-1, 3e-1]}


# --------------------------------------------------------------------------- E
def bc_stress(data, device, epochs=1):
    """Section 3 says the missing correction hurts most when beta2 is close to 1.
    Sweep beta2 in {0.99, 0.999, 0.9999} x lr, with / without correction."""
    res = []
    for b2 in (0.99, 0.999, 0.9999):
        for lr in (1e-3, 3e-3, 1e-2):
            for bc in (True, False):
                set_seed(0)
                net = mlp().to(device)
                opt = Adam(net.parameters(), lr=lr, betas=(0.9, b2), bias_correction=bc)
                h = train_classifier(net, opt, data, epochs=epochs, batch_size=128, seed=0)
                tl, ta = evaluate(net, data[2], data[3])
                r = dict(beta2=b2, lr=lr, bias_correction=bc, test_acc=ta,
                         max_loss_first200=float(np.max(h["step_loss"][:200])),
                         mean_loss_first200=float(np.mean(h["step_loss"][:200])),
                         diverged=h["diverged"])
                res.append(r)
                print(f"[E] beta2={b2:<6g} lr={lr:<6g} bc={bc!s:5s} test_acc={ta:.4f} "
                      f"max_loss_first200={r['max_loss_first200']:.2f}")
    return res


def mnist_sweep(data, device, epochs=3):
    res = {}
    for name, lrs in GRID.items():
        for lr in lrs:
            set_seed(0)
            net = mlp().to(device)
            opt = build(name, net.parameters(), lr)
            with Timer() as tm:
                h = train_classifier(net, opt, data, epochs=epochs, batch_size=128, seed=0)
            tl, ta = evaluate(net, data[2], data[3])
            res[f"{name}@{lr:g}"] = dict(opt=name, lr=lr, test_loss=tl, test_acc=ta,
                                         first100=float(np.mean(h["step_loss"][:100])),
                                         final_train=float(np.mean(h["step_loss"][-200:])),
                                         curve=h["step_loss"], diverged=h["diverged"], sec=tm.s)
            print(f"[C] {name:9s} lr={lr:<7g} test_acc={ta:.4f} train_loss(last200)="
                  f"{res[f'{name}@{lr:g}']['final_train']:.4f} first100={res[f'{name}@{lr:g}']['first100']:.3f} ({tm.s:.1f}s)")
    return res


# --------------------------------------------------------------------------- D
def quadratic(steps=3000, d=100, noise=1.0, seeds=3):
    """Noisy quadratic: f(x) = 1/2 sum h_i x_i^2, g = h*x + noise*N(0,1)*sqrt(h).
    Curvatures h_i are log-spaced over 3 decades (condition number 1e3)."""
    h = torch.logspace(-3, 0, d)
    grid = {"adam": [1e-3, 3e-3, 1e-2, 3e-2], "adam_nobc": [1e-3, 3e-3, 1e-2, 3e-2],
            "rmsprop": [1e-3, 3e-3, 1e-2, 3e-2], "sgdm": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]}
    res = {}
    for name, lrs in grid.items():
        best = None
        for lr in lrs:
            finals, curves = [], []
            for s in range(seeds):
                gen = torch.Generator().manual_seed(s)
                x = nn.Parameter(torch.ones(d) * 3.0)
                opt = build(name, [x], lr)
                curve = []
                for _ in range(steps):
                    x.grad = h * x.detach() + noise * h.sqrt() * torch.randn(d, generator=gen)
                    opt.step()
                    curve.append(0.5 * (h * x.detach() ** 2).sum().item())
                finals.append(np.mean(curve[-300:]))
                curves.append(curve)
            score = float(np.mean(finals))
            if not np.isfinite(score):
                score = float("inf")
            if best is None or score < best["final"]:
                best = dict(lr=lr, final=score, curve=np.mean(curves, 0).tolist())
        res[name] = best
        print(f"[D] {name:9s} best lr={best['lr']:<6g} mean f over last 300 steps={best['final']:.4f}")
    return res


def main():
    device = get_device()
    print("device:", device)
    data = load("mnist", device)

    diff = parity(data, device)
    ss = step_sizes(data, device)
    sweep = mnist_sweep(data, device)
    quad = quadratic()
    stress = bc_stress(data, device)

    # ---- pick best LR per optimiser (by test accuracy) for C
    best = {}
    for k, r in sweep.items():
        if r["opt"] not in best or r["test_acc"] > best[r["opt"]]["test_acc"]:
            best[r["opt"]] = r
    table = [{"optimizer": o, "best_lr": r["lr"], "test_acc": r["test_acc"],
              "mean_loss_first100": r["first100"], "train_loss_last200": r["final_train"]}
             for o, r in best.items()]
    full = [{k: v for k, v in r.items() if k != "curve"} for r in sweep.values()]
    save_json({"parity_max_abs_diff": diff, "mnist_best": table, "mnist_all": full,
               "quadratic": {k: {"lr": v["lr"], "final": v["final"]} for k, v in quad.items()},
               "bias_correction_stress": stress,
               "step_ratio_peak_measured": max(ss["measured_ratio"]),
               "step_ratio_peak_theory": max(ss["theory_ratio"])},
              os.path.join(RES, "results.json"))

    # ---- figures
    names = {"adam": "Adam", "adam_nobc": "Adam, no bias corr.", "rmsprop": "RMSProp",
             "sgdm": "SGD + momentum"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    t = np.arange(1, len(ss["theory_ratio"]) + 1)
    ax[0].plot(t, ss["measured_ratio"], label="measured (MLP, lr=1e-3)", lw=1)
    ax[0].plot(t, ss["theory_ratio"], "k--", label=r"theory $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$")
    ax[0].axhline(1, color="grey", lw=0.5)
    ax[0].set_xscale("log"); ax[0].set_xlabel("step t"); ax[0].set_ylabel("step size ratio")
    ax[0].set_title("A. Without bias correction, early steps are larger"); ax[0].legend()
    k = 25
    for o, r in best.items():
        c = np.convolve(r["curve"], np.ones(k) / k, mode="valid")
        ax[1].plot(c, label=f"{names[o]} (lr={r['lr']:g})", lw=1)
    ax[1].set_yscale("log"); ax[1].set_xlabel("step"); ax[1].set_ylabel("train loss (25-step avg)")
    ax[1].set_title("B. MNIST MLP, best LR per optimiser"); ax[1].legend()
    for o, r in quad.items():
        ax[2].plot(r["curve"], label=f"{names[o]} (lr={r['lr']:g})", lw=1)
    ax[2].set_yscale("log"); ax[2].set_xlabel("step"); ax[2].set_ylabel("f(x)")
    ax[2].set_title("C. Noisy quadratic, cond. no. 1e3"); ax[2].legend()
    savefig(fig, os.path.join(FIG, "adam_ablation.png"))

    # Bias-correction stress test: test accuracy after 1 epoch vs (beta2, lr)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for i, b2 in enumerate((0.99, 0.999, 0.9999)):
        for bc, sty in ((True, "o-"), (False, "s--")):
            rr = [r for r in stress if r["beta2"] == b2 and r["bias_correction"] == bc]
            ax[i].plot([r["lr"] for r in rr], [r["test_acc"] for r in rr], sty,
                       label="with correction" if bc else "without")
        ax[i].set_xscale("log"); ax[i].set_xlabel("learning rate")
        ax[i].set_title(rf"$\beta_2$ = {b2}")
    ax[0].set_ylabel("test accuracy after 1 epoch"); ax[0].legend()
    savefig(fig, os.path.join(FIG, "bias_correction_stress.png"))

    # ---- summary.csv
    record(PAPER, "parity_max_abs_diff_vs_torch", diff, "200 steps, same init")
    record(PAPER, "step_ratio_peak_nobc_over_bc", float(max(ss["measured_ratio"])), "measured, beta2=0.999")
    for o, r in best.items():
        record(PAPER, f"mnist_test_acc_{o}", r["test_acc"], f"best lr {r['lr']:g}, 3 epochs")
    for o, r in quad.items():
        record(PAPER, f"quadratic_final_f_{o}", r["final"], f"best lr {r['lr']:g}")
    for r in stress:
        if r["beta2"] == 0.9999 and r["lr"] == 1e-2:
            record(PAPER, f"stress_b2_0.9999_lr_0.01_acc_{'bc' if r['bias_correction'] else 'nobc'}",
                   r["test_acc"], "1 epoch")


if __name__ == "__main__":
    main()
