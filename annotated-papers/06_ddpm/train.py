"""Paper 06 ablation: linear vs cosine beta schedule, and how sample quality
falls as the number of ancestral sampling steps shrinks.

Author: Abyn Scaria.     Run:  python3 -u 06_ddpm/train.py [--steps 5000] [--n_samples 500]

Quality metric: an FID-PROXY, NOT FID. FID (Heusel et al. 2017) is the Frechet
distance between Inception-v3 pool features of real and generated images. Here
the feature extractor is a small CNN trained on MNIST in this script (64-d
penultimate layer), so absolute numbers are not comparable with any published
FID. They are only comparable with each other and with the real-vs-real floor
reported next to them.
"""
import argparse
import os
import sys

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import load
from common.harness import (get_device, set_seed, batches, train_classifier, evaluate,
                            record, save_json, savefig, Timer)
from model import Diffusion, TinyUNet, linear_betas, cosine_betas

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, RES = os.path.join(HERE, "figures"), os.path.join(HERE, "results")
PAPER = "06_ddpm"
SAMPLE_STEPS = [1000, 200, 50, 20, 10]
RAW_STEPS = [1000, 50]


# --------------------------------------------------------------------------- FID proxy
class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Flatten(), nn.Linear(64 * 7 * 7, 64), nn.ReLU())
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        return self.head(self.body(x))

    @torch.no_grad()
    def features(self, x01):
        self.eval()
        return torch.cat([self.body(xb) for xb, _ in batches(x01, None, 500, shuffle=False)]).cpu().double().numpy()


def frechet(a, b):
    """d^2 = ||mu_a - mu_b||^2 + Tr(S_a + S_b - 2 (S_a S_b)^{1/2})  (Frechet / 2-Wasserstein
    distance between Gaussians fitted to the two feature sets, as in FID)."""
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = scipy.linalg.sqrtm(s1 @ s2, disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


# --------------------------------------------------------------------------- training
def train_ddpm(betas, x01, steps, device, bs=128, ema_decay=0.999):
    set_seed(0)
    diff = Diffusion(betas, device)
    net = TinyUNet().to(device)
    ema = TinyUNet().to(device)
    ema.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)      # DDPM's CIFAR10 learning rate
    x = x01 * 2 - 1                                         # data scaled to [-1, 1] (section 4)
    g = torch.Generator().manual_seed(0)
    losses, s = [], 0
    while s < steps:
        for xb, _ in batches(x, None, bs, generator=g):
            loss = diff.loss(net, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():                           # EMA of weights, as in the paper
                for pe, p in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(p, 1 - ema_decay)
            losses.append(loss.item())
            s += 1
            if s % 500 == 0:
                print(f"    step {s}: loss(avg last 500) {np.mean(losses[-500:]):.4f}", flush=True)
            if s >= steps:
                break
    return diff, ema.eval(), losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--n_samples", type=int, default=500)
    a = ap.parse_args()
    device = get_device()
    print("device:", device)
    data = load("mnist", device)
    xtr, ytr, xte, yte = data
    n = a.n_samples

    # 1) feature network for the FID proxy
    set_seed(0)
    fnet = FeatureNet().to(device)
    train_classifier(fnet, torch.optim.Adam(fnet.parameters(), 1e-3), data, epochs=2, seed=0)
    clf_acc = evaluate(fnet, xte, yte)[1]
    print(f"[proxy] feature-net test accuracy {clf_acc:.4f}")
    f_test = fnet.features(xte[:5000])
    floor = frechet(fnet.features(xtr[:n]), f_test)                        # real vs real, same n
    noisy = frechet(fnet.features((xtr[:n] + 0.3 * torch.randn_like(xtr[:n])).clamp(0, 1)), f_test)
    print(f"[proxy] FD real(train,n={n}) vs test: {floor:.2f}   real+N(0,0.3^2) vs test: {noisy:.2f}")

    # 2) DDPM for both schedules, sampled with several step counts
    out = {"proxy": {"clf_acc": clf_acc, "floor_real_vs_real": floor, "real_plus_noise_0.3": noisy,
                     "n_samples": n}, "runs": {}}
    grids = {}
    for name, betas in (("linear", linear_betas()), ("cosine", cosine_betas())):
        print(f"[train] {name} schedule, {a.steps} steps")
        with Timer() as tm:
            diff, ema, losses = train_ddpm(betas, xtr, a.steps, device)
        os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
        torch.save(ema.state_dict(), os.path.join(HERE, "checkpoints", f"ema_{name}.pt"))
        r = {"train_sec": tm.s, "final_loss": float(np.mean(losses[-500:])), "by_steps": {}, "raw": {}}
        # clipped-x0 sampler at every step count; literal eq. (11) sampler at two of them
        for clip, ks, key in ((True, SAMPLE_STEPS, "by_steps"), (False, RAW_STEPS, "raw")):
            for k in ks:
                torch.manual_seed(123)
                with Timer() as ts:
                    xs = diff.sample(ema, n, steps=k, clip_x0=clip)
                x01 = (xs + 1) / 2
                fd = frechet(fnet.features(x01), f_test)
                with torch.no_grad():
                    pred = torch.cat([fnet(xb).argmax(1) for xb, _ in batches(x01, None, 500, shuffle=False)])
                hist = torch.bincount(pred, minlength=10).float() / n
                ent = float(-(hist[hist > 0] * hist[hist > 0].log()).sum() / np.log(10))
                r[key][k] = {"fd_proxy": fd, "class_entropy_norm": ent, "sample_sec": ts.s}
                if clip:
                    grids[(name, k)] = x01[:64].cpu().numpy()[:, 0]
                else:
                    grids[(name + "_raw", k)] = x01[:64].cpu().numpy()[:, 0]
                print(f"  {name:6s} {'clip-x0' if clip else 'eq.(11)'} sampling steps {k:4d}: "
                      f"FD-proxy {fd:8.2f}  class-entropy {ent:.3f}  ({ts.s:.0f}s)")
        r["loss_curve"] = losses
        out["runs"][name] = r
    save_json({k: v for k, v in out.items()}, os.path.join(RES, "results.json"))

    # ---- figure 1: FD-proxy vs sampling steps
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name in ("linear", "cosine"):
        bs_ = out["runs"][name]["by_steps"]
        line, = ax[0].plot(SAMPLE_STEPS, [bs_[k]["fd_proxy"] for k in SAMPLE_STEPS], "o-",
                           label=f"{name}, clipped-x0 sampler")
        rw = out["runs"][name]["raw"]
        ax[0].plot(RAW_STEPS, [rw[k]["fd_proxy"] for k in RAW_STEPS], "x--", color=line.get_color(),
                   label=f"{name}, literal eq. (11) sampler")
    ax[0].axhline(floor, color="green", ls="--", label=f"real vs real floor ({floor:.1f})")
    ax[0].axhline(noisy, color="grey", ls=":", label=f"real + N(0, 0.3^2) noise ({noisy:.1f})")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("ancestral sampling steps (respaced)"); ax[0].set_ylabel("FD-proxy (lower is better)")
    ax[0].set_title("A. Sample quality vs sampling steps"); ax[0].legend()
    T = 1000
    for name, betas in (("linear", linear_betas()), ("cosine", cosine_betas())):
        ax[1].plot(np.arange(1, T + 1) / T, torch.cumprod(1 - betas, 0).numpy(), label=name)
    ax[1].set_xlabel("t / T"); ax[1].set_ylabel(r"$\bar\alpha_t$ (signal fraction)")
    ax[1].set_title(r"B. $\bar\alpha_t$ under the two schedules"); ax[1].legend()
    savefig(fig, os.path.join(FIG, "ddpm_ablation.png"))

    # ---- figure 2: sample grids
    fig, axes = plt.subplots(2, len(SAMPLE_STEPS), figsize=(3 * len(SAMPLE_STEPS), 6.4))
    for i, name in enumerate(("linear", "cosine")):
        for j, k in enumerate(SAMPLE_STEPS):
            g = grids[(name, k)][:36].reshape(6, 6, 28, 28).transpose(0, 2, 1, 3).reshape(168, 168)
            axes[i, j].imshow(g, cmap="gray"); axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            axes[i, j].set_title(f"{name}, {k} steps\nFD-proxy {out['runs'][name]['by_steps'][k]['fd_proxy']:.1f}")
    savefig(fig, os.path.join(FIG, "ddpm_samples.png"))

    # ---- figure 3: the forward process q(x_t | x_0) on one digit
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.3))
    ts = [0, 50, 100, 200, 400, 600, 800, 999]
    for i, (name, betas) in enumerate((("linear", linear_betas()), ("cosine", cosine_betas()))):
        d = Diffusion(betas, device)
        torch.manual_seed(0)
        x0 = xte[:1] * 2 - 1
        eps = torch.randn_like(x0)
        for j, t in enumerate(ts):
            xt = d.q_sample(x0, torch.tensor([t], device=device), eps)
            axes[i, j].imshow(((xt[0, 0] + 1) / 2).clamp(0, 1).cpu(), cmap="gray")
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            axes[i, j].set_title(f"{name} t={t + 1}", fontsize=8)
    savefig(fig, os.path.join(FIG, "forward_process.png"))

    record(PAPER, "fd_proxy_floor_real_vs_real", floor, f"n={n}, 64-d MNIST CNN features")
    record(PAPER, "proxy_feature_net_test_acc", clf_acc)
    for name in ("linear", "cosine"):
        for k in SAMPLE_STEPS:
            record(PAPER, f"fd_proxy_{name}_{k}steps", out["runs"][name]["by_steps"][k]["fd_proxy"],
                   f"clipped-x0 sampler, {a.steps} train steps, n={n}")
        for k in RAW_STEPS:
            record(PAPER, f"fd_proxy_{name}_{k}steps_eq11", out["runs"][name]["raw"][k]["fd_proxy"],
                   "literal eq. (11) sampler")


if __name__ == "__main__":
    main()
