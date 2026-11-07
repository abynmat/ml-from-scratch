"""Paper 05 ablation: saturating vs non-saturating generator loss, measured by
mode coverage on a ring of 8 Gaussians.

Author: Abyn Scaria.     Run:  python3 -u 05_gan/train.py [--steps 8000] [--seeds 5]

Grid: {saturating, non-saturating} x {no head start, D pre-trained 2000 steps} x 5 seeds,
Adam for both players. The head-start condition recreates the situation section 3
of the paper warns about (paraphrased: early on, a poor G is rejected by D with
high confidence), which is exactly where log(1 - D(G(z))) saturates.
A small SGD probe (seed 0) checks whether Adam's gradient rescaling hides the effect.
Runs on CPU on purpose: 2-D MLPs are faster there than on a GPU.
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.harness import set_seed, record, save_json, savefig, Timer
from model import (Generator, Discriminator, d_loss, g_loss_saturating, g_loss_nonsaturating,
                   ring_of_gaussians, mode_stats)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, RES = os.path.join(HERE, "figures"), os.path.join(HERE, "results")
PAPER = "05_gan"
G_LOSS = {"saturating": g_loss_saturating, "non-saturating": g_loss_nonsaturating}
HEAD_STARTS = (0, 2000)


def make_opt(params, kind):
    if kind == "adam":
        return torch.optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))
    return torch.optim.SGD(params, lr=0.05)          # plain SGD: no gradient rescaling


def run(loss_name, head_start, seed, steps, bs=256, eval_every=500, snap_at=(), opt="adam"):
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    G, D = Generator(), Discriminator()
    oG, oD = make_opt(G.parameters(), opt), make_opt(D.parameters(), opt)

    def d_step():
        real = ring_of_gaussians(bs, generator=gen)
        with torch.no_grad():
            fake = G(bs)
        loss = d_loss(D(real), D(fake))
        oD.zero_grad(); loss.backward(); oD.step()

    for _ in range(head_start):         # D learns to reject the untrained G first
        d_step()

    hist, snaps, early = [], {}, []
    for s in range(1, steps + 1):
        d_step()                         # Algorithm 1 with k = 1 discriminator step
        fake = G(bs)
        fake.retain_grad()
        logit = D(fake)
        lg = G_LOSS[loss_name](logit)
        oG.zero_grad(); lg.backward(); oG.step()
        if s <= 500:
            early.append((torch.sigmoid(logit).mean().item(), fake.grad.norm(dim=1).mean().item() * bs))
        if s % eval_every == 0:
            with torch.no_grad():
                ms = mode_stats(G(2000))
            hist.append((s, ms["modes"], ms["hq_frac"], ms["kl_uniform"]))
        if s in snap_at:
            with torch.no_grad():
                snaps[s] = G(1500).numpy()
    with torch.no_grad():
        final = mode_stats(G(10000))
        final["D_of_G_end"] = torch.sigmoid(D(G(2000))).mean().item()
    return dict(hist=hist, snaps=snaps, early=early, final=final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    snap_at = (500, 2000, a.steps)
    runs = {}
    for loss_name in G_LOSS:
        for hs in HEAD_STARTS:
            for seed in range(a.seeds):
                with Timer() as tm:
                    r = run(loss_name, hs, seed, a.steps, snap_at=snap_at if seed == 0 else ())
                runs[(loss_name, hs, seed)] = r
                f = r["final"]
                print(f"[{loss_name:14s} head_start={hs:3d} seed={seed}] modes {f['modes']}/8  "
                      f"HQ {f['hq_frac']:.3f}  KL(uniform) {f['kl_uniform']:.3f}  "
                      f"early mean D(G(z)) {np.mean([e[0] for e in r['early'][:50]]):.3f}  "
                      f"early |dL/dx_fake| {np.mean([e[1] for e in r['early'][:50]]):.2e}  ({tm.s:.0f}s)")

    # ---- aggregate
    agg = []
    for loss_name in G_LOSS:
        for hs in HEAD_STARTS:
            fs = [runs[(loss_name, hs, s)]["final"] for s in range(a.seeds)]
            ea = [runs[(loss_name, hs, s)]["early"] for s in range(a.seeds)]
            agg.append(dict(loss=loss_name, head_start=hs,
                            modes_mean=float(np.mean([f["modes"] for f in fs])),
                            modes_per_seed=[f["modes"] for f in fs],
                            hq_mean=float(np.mean([f["hq_frac"] for f in fs])),
                            kl_mean=float(np.nanmean([f["kl_uniform"] for f in fs])),
                            early_D_of_G=float(np.mean([e[0] for r in ea for e in r[:50]])),
                            early_grad=float(np.mean([e[1] for r in ea for e in r[:50]]))))
    # ---- SGD probe: same grid, plain SGD, seed 0 only
    probe = []
    for loss_name in G_LOSS:
        for hs in HEAD_STARTS:
            r = run(loss_name, hs, 0, a.steps, opt="sgd")
            f = r["final"]
            early_d = float(np.mean([e[0] for e in r["early"][:50]]))
            early_g = float(np.mean([e[1] for e in r["early"][:50]]))
            probe.append(dict(loss=loss_name, head_start=hs, modes=f["modes"], hq_frac=f["hq_frac"],
                              D_of_G_end=f["D_of_G_end"], early_D_of_G=early_d, early_grad=early_g))
            print(f"[SGD probe {loss_name:14s} head_start={hs}] modes {f['modes']}/8 HQ {f['hq_frac']:.3f} "
                  f"early D(G(z)) {early_d:.3f} early grad {early_g:.2e} final D(G(z)) {f['D_of_G_end']:.3f}")

    save_json({"args": vars(a), "summary": agg, "sgd_probe": probe,
               "runs": {f"{k[0]}|hs{k[1]}|s{k[2]}": {"final": v["final"], "hist": v["hist"]}
                        for k, v in runs.items()}}, os.path.join(RES, "results.json"))

    # ---- figure 1: samples (seed 0) per condition over time
    real = ring_of_gaussians(1500, generator=torch.Generator().manual_seed(9)).numpy()
    conds = [(l, hs) for l in G_LOSS for hs in HEAD_STARTS]
    fig, ax = plt.subplots(len(conds), len(snap_at), figsize=(3.2 * len(snap_at), 3.0 * len(conds)))
    for i, (l, hs) in enumerate(conds):
        r = runs[(l, hs, 0)]
        for j, s in enumerate(snap_at):
            ax[i, j].scatter(real[:, 0], real[:, 1], s=2, c="lightgrey")
            ax[i, j].scatter(r["snaps"][s][:, 0], r["snaps"][s][:, 1], s=2, c="tab:blue")
            ax[i, j].set_xlim(-3, 3); ax[i, j].set_ylim(-3, 3)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(f"step {s}")
        ax[i, 0].set_ylabel(f"{l}\nD head start {hs}")
    savefig(fig, os.path.join(FIG, "gan_samples.png"))

    # ---- figure 2: mode coverage over training + early gradient magnitude
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for l, hs in conds:
        H = np.array([runs[(l, hs, s)]["hist"] for s in range(a.seeds)])  # seeds x evals x 4
        ax[0].plot(H[0, :, 0], H[:, :, 1].mean(0), "o-" if l == "non-saturating" else "s--",
                   label=f"{l}, head start {hs}")
        E = np.array([[e[1] for e in runs[(l, hs, s)]["early"]] for s in range(a.seeds)]).mean(0)
        ax[1].semilogy(np.convolve(E, np.ones(10) / 10, "valid"),
                       "-" if l == "non-saturating" else "--", label=f"{l}, head start {hs}")
    ax[0].set_xlabel("generator step"); ax[0].set_ylabel(f"modes covered (of 8, mean of {a.seeds} seeds)")
    ax[0].set_ylim(0, 8.5); ax[0].set_title("A. Mode coverage"); ax[0].legend()
    ax[1].set_xlabel("generator step"); ax[1].set_ylabel("||dL_G / dG(z)|| per sample")
    ax[1].set_title("B. Gradient reaching G in the first 500 steps"); ax[1].legend()
    savefig(fig, os.path.join(FIG, "gan_mode_coverage.png"))

    for r in agg:
        tag = f"{r['loss']}_hs{r['head_start']}"
        record(PAPER, f"modes_covered_{tag}", r["modes_mean"], f"mean of {a.seeds} seeds, of 8")
        record(PAPER, f"hq_frac_{tag}", r["hq_mean"], "within 3 std of a mode")
        record(PAPER, f"early_grad_{tag}", r["early_grad"], "first 50 G steps")


if __name__ == "__main__":
    main()
