# 05 Generative Adversarial Nets (Goodfellow et al., 2014) [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

## What the paper claims
- Eq. (1): min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]. Algorithm 1 alternates k discriminator steps with one generator step.
- Section 3: early in training, D rejects G's samples confidently, so log(1 - D(G(z))) saturates and the generator gets almost no gradient. The paper's fix is to train G to *maximise* log D(G(z)) instead. It has the same fixed point but much stronger gradients early on.

## Where it lives in the code
`model.py`: `d_loss`, `g_loss_saturating` (eq. 1 literally), `g_loss_nonsaturating` (the section 3 fix), written in logit form. The logit-gradient of each loss is spelled out in the docstrings. `ring_of_gaussians` and `mode_stats` are the benchmark and the metrics: a sample counts as high quality (HQ) if it lies within 3 std of a mode, and a mode counts as covered if it holds at least 1% of samples.

## What the ablation found (8 Gaussians, radius 2, std 0.02; 2x128 MLPs; 8,000 G steps; 5 seeds)
| G loss | D head start | Modes covered (mean, per seed) | HQ fraction | early D(G(z)) | early per-sample grad on G(z) |
|---|---|---|---|---|---|
| saturating | 0 | **7.8** (7,8,8,8,8) | 0.71 | 0.48 | 0.10 |
| non-saturating | 0 | 7.4 (7,8,8,6,8) | 0.81 | 0.48 | 0.11 |
| saturating | 2000 | 7.4 (8,5,8,8,8) | 0.60 | 0.07 | 0.40 |
| non-saturating | 2000 | **3.4** (2,3,4,4,4) | **0.84** | 0.04 | 5.30 |

All runs use Adam (lr 2e-4, betas 0.5/0.999). "Early" means the first 50 generator steps.

SGD probe (plain SGD lr 0.05, seed 0): every condition collapsed to 1 mode. With the D head start, the saturating loss's early gradient was 1.9e-3, against 0.97 for the non-saturating loss (about 500x smaller), and D(G(z)) sat at 0.000.

## Where the paper and this run disagree, and why
- **The mechanism checks out.** With a confident D, the saturating loss gives G a gradient roughly 100-500x smaller (see the first steps of `figures/gan_mode_coverage.png` B and the SGD probe), exactly as section 3 says.
- **The outcome, under Adam, is the opposite of the folklore.** Adam divides by a running RMS of the gradient, so a uniformly tiny gradient still produces a normal-sized step (see paper 01: Adam is invariant to gradient scale). The saturating generator escapes within about 30 steps and then covers 7.4/8 modes.
- **The non-saturating loss with a strong D causes mode collapse.** Its gradients are large (5.3 per sample) and point every fake toward wherever D is currently least confident. G pours into 2-4 modes with high precision (HQ 0.84) and never leaves. The paper's fix trades *vanishing* gradients for *over-confident* gradients. Arjovsky & Bottou ([arXiv:1701.04862](https://arxiv.org/abs/1701.04862)) analyse this, showing that the -log D objective has unstable, high-variance updates when D is near optimal.
- With a balanced start (no head start) the two losses are statistically indistinguishable at 5 seeds: 7.8 vs 7.4 modes.

## Caveats
2-D toy data, 5 seeds, one architecture, and one Adam setting. Plain SGD failed in every condition here, so the SGD probe only isolates the gradient-magnitude mechanism. It says nothing about final quality. The 1% / 3-std coverage thresholds are a convention (cf. VEEGAN / Unrolled-GAN evaluations), not something from the GAN paper.
