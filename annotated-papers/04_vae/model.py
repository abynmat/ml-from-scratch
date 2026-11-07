"""Variational autoencoder with the reparameterisation trick and a beta knob.

Papers
  Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014 (arXiv:1312.6114)
  Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained
      Variational Framework", ICLR 2017 (beta-weighted KL term)
Author: Abyn Scaria. Equation numbers refer to Kingma & Welling (v10 on arXiv).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """MLP encoder q_phi(z|x) = N(mu, diag sigma^2) and Bernoulli decoder p_theta(x|z)
    (the MNIST setup of the paper's section 5 and Appendix C, with wider layers)."""

    def __init__(self, z_dim=10, hidden=512):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.mu = nn.Linear(hidden // 2, z_dim)
        self.logvar = nn.Linear(hidden // 2, z_dim)   # predict log sigma^2 for stability
        self.dec = nn.Sequential(nn.Linear(z_dim, hidden // 2), nn.ReLU(),
                                 nn.Linear(hidden // 2, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 784))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def decode_logits(self, z):
        return self.dec(z).view(-1, 1, 28, 28)

    @staticmethod
    def reparameterize(mu, logvar):
        # Section 2.4 / eq. (10): z = mu + sigma (.) eps,  eps ~ N(0, I).
        # The randomness moves into eps, so dz/dmu and dz/dsigma exist and the
        # Monte Carlo estimate of the ELBO is differentiable w.r.t. phi.
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_logits(z), mu, logvar


def kl_per_dim(mu, logvar):
    """Closed-form -D_KL(q(z|x) || N(0, I)) from Appendix B, per latent dimension:
    -KL = 1/2 (1 + log sigma^2 - mu^2 - sigma^2)  -> we return +KL (>= 0)."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


def loss_terms(model, x):
    """Per-example reconstruction NLL and KL (nats), single-sample (L = 1) estimator.

    eq. (10):  L(theta, phi; x) ~= -KL(q || p) + log p_theta(x | z),  z = mu + sigma*eps
    With grey-level pixels in [0, 1] treated as Bernoulli means, -log p(x|z) is the
    binary cross-entropy summed over the 784 pixels.
    """
    logits, mu, logvar = model(x)
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=(1, 2, 3))
    kl = kl_per_dim(mu, logvar).sum(1)
    return rec, kl, mu, logvar


def beta_loss(model, x, beta):
    """beta-VAE objective (Higgins et al. 2017, eq. 4): E[log p(x|z)] - beta * KL.
    beta = 1 recovers the ELBO of eq. (3); we minimise its negative."""
    rec, kl, _, _ = loss_terms(model, x)
    return (rec + beta * kl).mean(), rec.mean(), kl.mean()
