"""DDPM: forward noising, epsilon-prediction loss and ancestral sampling.

Papers
  Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020
      (arXiv:2006.11239). Equation / algorithm numbers refer to this paper.
  Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021
      (arXiv:2102.09672): the cosine schedule (their eq. 17) and timestep
      respacing for fewer sampling steps (their section 4).
Author: Abyn Scaria.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- schedules
def linear_betas(T=1000):
    """Section 4 of Ho et al.: beta_1 = 1e-4 rising linearly to beta_T = 0.02."""
    return torch.linspace(1e-4, 0.02, T, dtype=torch.float64)


def cosine_betas(T=1000, s=0.008):
    """Nichol & Dhariwal eq. (17): alpha_bar(t) = f(t)/f(0), f(t) = cos^2(((t/T + s)/(1 + s)) * pi/2),
    beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, clipped at 0.999."""
    t = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    abar = f / f[0]
    return (1 - abar[1:] / abar[:-1]).clamp(max=0.999)


class Diffusion:
    """Holds the schedule and implements q(x_t | x_0), the loss and the sampler."""

    def __init__(self, betas, device):
        self.T = len(betas)
        self.betas = betas
        self.alphas = 1 - betas                          # alpha_t = 1 - beta_t
        self.abar = torch.cumprod(self.alphas, 0)        # alpha_bar_t = prod_{s<=t} alpha_s
        self.device = device
        f = lambda v: v.float().to(device)
        self.sqrt_abar, self.sqrt_1m_abar = f(self.abar.sqrt()), f((1 - self.abar).sqrt())

    def q_sample(self, x0, t, eps):
        """eq. (4): q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I),
        sampled in one shot as x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps."""
        a = self.sqrt_abar[t].view(-1, 1, 1, 1)
        b = self.sqrt_1m_abar[t].view(-1, 1, 1, 1)
        return a * x0 + b * eps

    def loss(self, model, x0):
        """Algorithm 1 / eq. (14), L_simple = E_{t, x0, eps} || eps - eps_theta(x_t, t) ||^2,
        t ~ Uniform{1..T} (0-indexed here)."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, eps), t), eps)

    @torch.no_grad()
    def sample(self, model, n, steps=None, shape=(1, 28, 28), clip_x0=True):
        """Algorithm 2 (ancestral sampling) with sigma_t^2 = beta_t.

        With `steps` < T we respace: pick an evenly spaced subsequence S of
        timesteps and use the betas that reproduce the SAME alpha_bar on S,
        beta'_i = 1 - abar_{S_i} / abar_{S_{i-1}} (Nichol & Dhariwal, section 4).
        The trained eps-model is reused unchanged; only the step sizes grow.

        clip_x0=False: the literal Algorithm 2 / eq. (11) update.
        clip_x0=True : the algebraically identical posterior-mean form, eq. (7),
            mu~_t(x_t, x0_hat) with x0_hat = (x_t - sqrt(1-abar_t) eps) / sqrt(abar_t)
            clipped to the data range [-1, 1]. This is what the paper's released
            code does ("clip_denoised"). It matters when beta_t is near 1 (the
            cosine schedule clips beta_T at 0.999), where eq. (11) divides by
            sqrt(alpha_t) ~ 0.03 and amplifies any eps error about 30x.
        """
        if clip_x0:
            return self._sample_clipped(model, n, steps, shape)
        steps = steps or self.T
        S = torch.linspace(0, self.T - 1, steps).round().long()
        abar = self.abar[S]
        abar_prev = torch.cat([torch.ones(1, dtype=abar.dtype), abar[:-1]])
        betas = 1 - abar / abar_prev
        x = torch.randn(n, *shape, device=self.device)                # x_T ~ N(0, I)
        for i in reversed(range(steps)):
            t = torch.full((n,), int(S[i]), device=self.device, dtype=torch.long)
            b, ab = float(betas[i]), float(abar[i])
            eps = model(x, t)
            # eq. (11) / Algorithm 2 line 4:
            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - abar_t) * eps_theta) + sigma_t z
            mean = (x - b / math.sqrt(1 - ab) * eps) / math.sqrt(1 - b)
            x = mean + (math.sqrt(b) * torch.randn_like(x) if i > 0 else 0)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def _sample_clipped(self, model, n, steps, shape):
        steps = steps or self.T
        S = torch.linspace(0, self.T - 1, steps).round().long()
        abar = self.abar[S]
        abar_prev = torch.cat([torch.ones(1, dtype=abar.dtype), abar[:-1]])
        betas = 1 - abar / abar_prev
        x = torch.randn(n, *shape, device=self.device)
        for i in reversed(range(steps)):
            t = torch.full((n,), int(S[i]), device=self.device, dtype=torch.long)
            b, ab, abp = float(betas[i]), float(abar[i]), float(abar_prev[i])
            eps = model(x, t)
            # invert eq. (4) for x0, then clip to the data range
            x0 = ((x - math.sqrt(1 - ab) * eps) / math.sqrt(ab)).clamp(-1, 1)
            # eq. (7): mu~_t = sqrt(abar_{t-1}) beta_t / (1 - abar_t) x0 + sqrt(alpha_t)(1 - abar_{t-1}) / (1 - abar_t) x_t
            mean = (math.sqrt(abp) * b / (1 - ab)) * x0 + (math.sqrt(1 - b) * (1 - abp) / (1 - ab)) * x
            x = mean + (math.sqrt(b) * torch.randn_like(x) if i > 0 else 0)
        return x.clamp(-1, 1)


# --------------------------------------------------------------------------- eps-network
def timestep_embedding(t, dim):
    """Sinusoidal position embedding of t (the Transformer encoding, as DDPM uses)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], 1)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t = nn.Linear(t_dim, c_out)
        self.n2, self.c2 = nn.GroupNorm(8, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]          # inject the timestep
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """A deliberately small U-Net (28 -> 14 -> 7 and back, 32/64/64 channels,
    no attention), roughly 1M parameters. DDPM's CIFAR model is ~36M."""

    def __init__(self, ch=(32, 64, 64), t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        c1, c2, c3 = ch
        self.inp = nn.Conv2d(1, c1, 3, padding=1)
        self.d1 = ResBlock(c1, c1, t_dim)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)      # 28 -> 14
        self.d2 = ResBlock(c2, c2, t_dim)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)      # 14 -> 7
        self.mid = ResBlock(c3, c3, t_dim)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)  # 7 -> 14
        self.u2 = ResBlock(2 * c2, c2, t_dim)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)  # 14 -> 28
        self.u1 = ResBlock(2 * c1, c1, t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))
        h1 = self.d1(self.inp(x), temb)
        h2 = self.d2(self.down1(h1), temb)
        h = self.mid(self.down2(h2), temb)
        h = self.u2(torch.cat([self.up2(h), h2], 1), temb)
        h = self.u1(torch.cat([self.up1(h), h1], 1), temb)
        return self.out(h)
