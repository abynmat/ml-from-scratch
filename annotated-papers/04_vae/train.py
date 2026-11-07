"""Paper 04 ablation: the beta sweep. How does weighting the KL term trade
reconstruction against rate, and what happens to the latent code?

Author: Abyn Scaria.     Run:  python3 -u 04_vae/train.py [--epochs 8]

MNIST, MLP VAE with z_dim = 10, Adam 1e-3, batch 128. For each beta we report on
the test set: reconstruction NLL, KL ("rate"), the beta=1 negative ELBO, and the
number of active latent units (Burda et al. 2016: Cov_x(E_q[z_j]) > 0.01).
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
from common.data import load
from common.harness import get_device, set_seed, batches, record, save_json, savefig, Timer
from model import VAE, beta_loss, loss_terms, kl_per_dim

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, RES = os.path.join(HERE, "figures"), os.path.join(HERE, "results")
PAPER = "04_vae"
BETAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


@torch.no_grad()
def test_metrics(model, x):
    model.eval()
    rec, kl, mus, kld = [], [], [], []
    for xb, _ in batches(x, None, 1000, shuffle=False):
        r, k, mu, logvar = loss_terms(model, xb)
        rec.append(r); kl.append(k); mus.append(mu); kld.append(kl_per_dim(mu, logvar))
    rec, kl, mus, kld = map(torch.cat, (rec, kl, mus, kld))
    active = (mus.var(0) > 0.01).sum().item()
    return dict(rec=rec.mean().item(), kl=kl.mean().item(), neg_elbo=(rec + kl).mean().item(),
                active_units=int(active), kl_per_dim=kld.mean(0).tolist())


def train(beta, data, epochs, z_dim=10):
    xtr, _, xte, _ = data
    set_seed(0)
    model = VAE(z_dim).to(xtr.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        model.train()
        for xb, _ in batches(xtr, None, 128, generator=g):
            loss, _, _ = beta_loss(model, xb, beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model, test_metrics(model, xte)


@torch.no_grad()
def traversal(model, x0, dims, vals):
    """Decode mu(x0) with one latent dimension swept over `vals` at a time."""
    model.eval()
    mu, _ = model.encode(x0[None])
    rows = []
    for d in dims:
        z = mu.repeat(len(vals), 1)
        z[:, d] = torch.tensor(vals, dtype=torch.float32, device=z.device)
        rows.append(torch.sigmoid(model.decode_logits(z)).cpu().numpy()[:, 0])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    device = get_device()
    print("device:", device)
    data = load("mnist", device)
    xte = data[2]

    results, models = {}, {}
    for beta in BETAS:
        with Timer() as tm:
            m, met = train(beta, data, a.epochs)
        met["sec"] = tm.s
        results[beta], models[beta] = met, m
        print(f"[beta={beta:<5g}] rec {met['rec']:7.2f}  KL {met['kl']:6.2f}  -ELBO {met['neg_elbo']:7.2f}  "
              f"active units {met['active_units']}/10  ({tm.s:.0f}s)")
    save_json({"args": vars(a), "by_beta": {str(b): r for b, r in results.items()}},
              os.path.join(RES, "results.json"))

    # ---- figure 1: rate-distortion trade-off + active units
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    R = [results[b]["kl"] for b in BETAS]
    D = [results[b]["rec"] for b in BETAS]
    ax[0].plot(R, D, "o-")
    for b, r, d in zip(BETAS, R, D):
        ax[0].annotate(rf"$\beta$={b:g}", (r, d), textcoords="offset points", xytext=(5, 5))
    ax[0].set_xlabel("rate: KL(q(z|x) || p(z))  [nats]")
    ax[0].set_ylabel("distortion: -log p(x|z)  [nats]")
    ax[0].set_title("A. Reconstruction vs KL as beta varies (test set)")
    ax2 = ax[1]
    ax2.semilogx(BETAS, [results[b]["active_units"] for b in BETAS], "s-", color="tab:red")
    ax2.set_xlabel(r"$\beta$"); ax2.set_ylabel("active latent units (of 10)")
    ax2.set_title("B. Latent dimensions the encoder actually uses")
    ax2.set_ylim(0, 10.5)
    savefig(fig, os.path.join(FIG, "beta_tradeoff.png"))

    # ---- figure 2: latent traversals for beta = 1 and beta = 4
    vals = np.linspace(-3, 3, 9)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, beta in zip(axes, (1.0, 4.0)):
        kld = np.array(results[beta]["kl_per_dim"])
        dims = list(np.argsort(-kld)[:6])
        grid = traversal(models[beta], xte[0], dims, vals)
        img = np.concatenate([np.concatenate(list(r), axis=1) for r in grid], axis=0)
        ax.imshow(img, cmap="gray")
        ax.set_yticks([14 + 28 * i for i in range(len(dims))])
        ax.set_yticklabels([f"z{d} (KL {kld[d]:.1f})" for d in dims])
        ax.set_xticks([14 + 28 * i for i in range(len(vals))])
        ax.set_xticklabels([f"{v:+.1f}" for v in vals])
        ax.set_title(rf"$\beta$ = {beta:g}: traverse top-6 dims by KL")
    savefig(fig, os.path.join(FIG, "latent_traversals.png"))

    # ---- figure 3: reconstructions and prior samples per beta
    torch.manual_seed(1)
    zs = torch.randn(8, 10, device=device)
    fig, axes = plt.subplots(len(BETAS), 2, figsize=(9, 1.3 * len(BETAS)))
    with torch.no_grad():
        for i, b in enumerate(BETAS):
            m = models[b].eval()
            mu, _ = m.encode(xte[:8])
            rec = torch.sigmoid(m.decode_logits(mu)).cpu().numpy()[:, 0]
            smp = torch.sigmoid(m.decode_logits(zs)).cpu().numpy()[:, 0]
            axes[i, 0].imshow(np.concatenate(list(rec), 1), cmap="gray")
            axes[i, 1].imshow(np.concatenate(list(smp), 1), cmap="gray")
            axes[i, 0].set_ylabel(rf"$\beta$={b:g}", rotation=0, labelpad=25)
            for j in (0, 1):
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    axes[0, 0].set_title("reconstructions of test digits 0-7")
    axes[0, 1].set_title("samples z ~ N(0, I)")
    savefig(fig, os.path.join(FIG, "recon_and_samples.png"))

    for b in BETAS:
        r = results[b]
        record(PAPER, f"beta{b:g}_rec_nats", r["rec"], "test, z=10")
        record(PAPER, f"beta{b:g}_kl_nats", r["kl"], "test, z=10")
        record(PAPER, f"beta{b:g}_active_units", r["active_units"], "of 10")
    record(PAPER, "beta1_neg_elbo_nats", results[1.0]["neg_elbo"], "grey-level MNIST, not binarized")


if __name__ == "__main__":
    main()
