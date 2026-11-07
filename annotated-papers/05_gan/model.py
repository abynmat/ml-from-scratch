"""A two-player GAN on 2-D data, with both generator losses from the paper.

Paper: Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014
       (arXiv:1406.2661). Equation numbers refer to that paper.
Author: Abyn Scaria.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(d_in, d_out, hidden=128):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, d_out))


class Generator(nn.Module):
    """G(z; theta_g): maps noise z ~ p_z = N(0, I_2) to a point in R^2."""

    def __init__(self, z_dim=2):
        super().__init__()
        self.z_dim = z_dim
        self.net = mlp(z_dim, 2)

    def forward(self, n):
        z = torch.randn(n, self.z_dim, device=next(self.parameters()).device)
        return self.net(z)


class Discriminator(nn.Module):
    """D(x; theta_d) = sigmoid(logit). We keep logits for numerical stability."""

    def __init__(self):
        super().__init__()
        self.net = mlp(2, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# eq. (1): min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
# With D = sigmoid(l):  log D = -softplus(-l),   log(1 - D) = -softplus(l).

def d_loss(d_real_logit, d_fake_logit):
    """D ascends V: maximise log D(x) + log(1 - D(G(z))) (Algorithm 1, D step)."""
    return F.softplus(-d_real_logit).mean() + F.softplus(d_fake_logit).mean()


def g_loss_saturating(d_fake_logit):
    """The minimax generator objective taken literally from eq. (1):
    minimise log(1 - D(G(z))) = -softplus(l). Its gradient w.r.t. the logit is
    -sigmoid(l), which goes to 0 exactly when D confidently rejects fakes (l << 0)."""
    return -F.softplus(d_fake_logit).mean()


def g_loss_nonsaturating(d_fake_logit):
    """Section 3's fix: "train G to maximize log D(G(z))", i.e. minimise
    -log D = softplus(-l). Gradient w.r.t. the logit is -(1 - sigmoid(l)), which
    is LARGE when D rejects fakes. Same fixed point, stronger early gradients."""
    return F.softplus(-d_fake_logit).mean()


def ring_of_gaussians(n, k=8, radius=2.0, std=0.02, generator=None):
    """Mixture of k isotropic Gaussians evenly spaced on a circle (the standard
    mode-collapse benchmark, cf. Metz et al. 2017 'Unrolled GANs')."""
    idx = torch.randint(0, k, (n,), generator=generator)
    ang = 2 * math.pi * idx.float() / k
    centers = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], 1)
    return centers + std * torch.randn(n, 2, generator=generator)


def mode_stats(x, k=8, radius=2.0, std=0.02, min_frac=0.01):
    """Mode coverage on the ring.
    A sample is 'high quality' if it lies within 3 std of its nearest mode centre.
    A mode is 'covered' if it holds at least `min_frac` of all samples.
    Also returns KL(mode histogram of HQ samples || uniform), 0 = perfectly balanced."""
    ang = 2 * math.pi * torch.arange(k).float() / k
    centers = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], 1)
    d = torch.cdist(x.cpu(), centers)
    dist, nearest = d.min(1)
    hq = dist < 3 * std
    counts = torch.bincount(nearest[hq], minlength=k).float()
    covered = int((counts >= min_frac * x.shape[0]).sum())
    p = counts / counts.sum().clamp(min=1)
    kl = float((p[p > 0] * (p[p > 0] * k).log()).sum()) if counts.sum() > 0 else float("nan")
    return dict(modes=covered, hq_frac=float(hq.float().mean()), kl_uniform=kl)
