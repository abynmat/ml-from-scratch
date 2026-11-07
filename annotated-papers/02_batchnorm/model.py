"""Batch Normalization and Layer Normalization written from their equations.

Papers
  Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by
      Reducing Internal Covariate Shift", ICML 2015 (arXiv:1502.03167)
  Ba, Kiros & Hinton, "Layer Normalization", 2016 (arXiv:1607.06450)
Author: Abyn Scaria.
"""
import torch
import torch.nn as nn


class BatchNorm2d(nn.Module):
    """BN for conv feature maps.

    For conv layers the paper normalises each channel jointly over the batch
    AND spatial positions, so the effective mini-batch is m = N*H*W (section 3.2).
    """

    def __init__(self, c, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.gamma = nn.Parameter(torch.ones(c))     # learned scale (Algorithm 1)
        self.beta = nn.Parameter(torch.zeros(c))     # learned shift (Algorithm 1)
        # Population statistics used at inference (Algorithm 2, steps 8-11).
        self.register_buffer("running_mean", torch.zeros(c))
        self.register_buffer("running_var", torch.ones(c))

    def forward(self, x):
        if self.training:
            # Algorithm 1:  mu_B = 1/m sum x_i ;  sigma_B^2 = 1/m sum (x_i - mu_B)^2
            mu = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                m = x.numel() / x.shape[1]
                # Algorithm 2, step 10: Var[x] <- m/(m-1) * E_B[sigma_B^2]
                # (an exponential moving average stands in for the average over batches)
                self.running_mean.lerp_(mu, self.momentum)
                self.running_var.lerp_(var * m / (m - 1), self.momentum)
        else:
            # Algorithm 2, step 11: inference uses fixed population statistics,
            # so the output for one example no longer depends on its batch-mates.
            mu, var = self.running_mean, self.running_var
        # Algorithm 1: x_hat = (x - mu_B) / sqrt(sigma_B^2 + eps);  y = gamma x_hat + beta
        x_hat = (x - mu[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]


class LayerNorm2d(nn.Module):
    """LN for conv maps: statistics over all hidden units of ONE example (eq. 3),
    i.e. over (C, H, W); per-channel gain and bias as in eq. (5). Identical in
    train and eval mode, so it has no running statistics at all."""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        # eq. (3): mu^l = 1/H sum_i a_i ;  sigma^l = sqrt(1/H sum_i (a_i - mu^l)^2)
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        # eq. (5): h = g / sigma * (a - mu) + b
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return self.gain[None, :, None, None] * x_hat + self.bias[None, :, None, None]


def norm_layer(kind, c):
    if kind == "bn":
        return BatchNorm2d(c)
    if kind == "ln":
        return LayerNorm2d(c)
    return nn.Identity()


class CNN(nn.Module):
    """Six 3x3 conv layers (32-32-64-64-128-128), two max-pools, global average pool.
    Normalisation goes between each conv and its ReLU, as in the BN paper (eq. in
    section 3.2: z = g(BN(Wu)), the conv bias is dropped because BN's beta replaces it)."""

    def __init__(self, norm="bn", n_classes=10):
        super().__init__()
        chans = [1, 32, 32, 64, 64, 128, 128]
        layers = []
        for i in range(6):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=(norm == "none")),
                       norm_layer(norm, chans[i + 1]), nn.ReLU()]
            if i in (1, 3):
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))
