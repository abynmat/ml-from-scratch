# 04 Variational Autoencoder (Kingma & Welling, 2013) + beta-VAE (Higgins et al., 2017)

[VAE arXiv:1312.6114](https://arxiv.org/abs/1312.6114) · [beta-VAE, ICLR 2017](https://openreview.net/forum?id=Sy2fzU9gl)

## What the papers claim
- VAE eq. (3): log p(x) >= L = -KL(q(z|x) || p(z)) + E_q[log p(x|z)], the ELBO.
- Section 2.4 / eq. (10): write z = mu + sigma * eps with eps ~ N(0, I). The single-sample Monte Carlo ELBO is then differentiable in the encoder parameters, and with a Gaussian prior the KL has the closed form in Appendix B.
- beta-VAE (eq. 4): multiply the KL by beta > 1. The paper claims this gives more disentangled factors at the cost of reconstruction quality.

## Where it lives in the code
`model.py`: `VAE.reparameterize` (eq. 10), `kl_per_dim` (Appendix B), `loss_terms` (single-sample ELBO), `beta_loss` (beta-VAE eq. 4).

## What the ablation found (MNIST, z = 10, 8 epochs, test set)
| beta | reconstruction -log p(x\|z) [nats] | KL [nats] | -ELBO (beta=1 bound) | active units /10 |
|---|---|---|---|---|
| 0.25 | 81.95 | 26.62 | 108.57 | 10 |
| 0.5 | 83.84 | 22.30 | **106.15** | 10 |
| 1 | 89.72 | 16.96 | 106.69 | 9 |
| 2 | 99.45 | 12.13 | 111.58 | 9 |
| 4 | 115.78 | 7.74 | 123.52 | 6 |
| 8 | 138.38 | 4.39 | 142.77 | 5 |
| 16 | 181.51 | 1.14 | 182.65 | 2 |

Figures: `figures/beta_tradeoff.png` (rate-distortion curve and active units), `figures/latent_traversals.png`, `figures/recon_and_samples.png`.

## Where the paper and this run disagree, and why
- **The trade-off is monotone and steep.** Each doubling of beta removes roughly 30-45% of the KL and adds 6-43 nats of reconstruction error. It traces a rate-distortion curve (cf. Alemi et al. 2018, "Fixing a Broken ELBO").
- **beta < 1 gave the best bound, slightly.** beta = 0.5 reached -ELBO 106.15 vs 106.69 at beta = 1. This should not happen if beta = 1 were optimising the bound perfectly. It points to amortisation and optimisation slack after only 8 epochs, not to a flaw in the bound. The gap is 0.5 nats from one seed, so it may well be noise.
- **"Disentanglement" mostly shows up here as compression.** Higher beta shuts latent units off (10 -> 2 active units). In the traversals, each beta = 4 dimension does one cleaner thing (e.g. 7 -> 0, slant, 7 -> 4/9). At beta = 1 more dimensions mix stroke-thickness and identity changes. That fits the paper's claim, but it is a visual judgement on MNIST, which has no ground-truth factors. The paper measured disentanglement on dSprites with a supervised metric, which this repo does not reproduce.

## Caveats
Grey-level pixels in [0, 1] are treated as Bernoulli means, so reconstruction NLL has a non-zero floor (the pixel entropy). That means these nats are **not** comparable to binarised-MNIST log-likelihoods in the literature. One seed and a single-sample ELBO, with no importance-weighted bound.
