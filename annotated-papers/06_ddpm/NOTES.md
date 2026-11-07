# 06 Denoising Diffusion Probabilistic Models (Ho et al., 2020) [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

Cosine schedule and respacing: Nichol & Dhariwal, 2021, [arXiv:2102.09672](https://arxiv.org/abs/2102.09672).

## What the papers claim
- Eq. (2): the forward process adds Gaussian noise with variances beta_1..beta_T. Eq. (4): it has the closed form q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I), so you can jump straight to any t.
- Eq. (11) + eq. (14): parameterise the reverse mean through a noise predictor eps_theta. The weighted bound then reduces to L_simple = ||eps - eps_theta(x_t, t)||^2 (Algorithm 1). Sampling is Algorithm 2, T = 1000 ancestral steps.
- Nichol & Dhariwal: the linear schedule wastes many steps at the noisy end and destroys information too fast on small images. The cosine schedule (their eq. 17) fixes this, and it degrades more gracefully when you sample with fewer steps.

## Where it lives in the code
`model.py`: `linear_betas`, `cosine_betas`, `Diffusion.q_sample` (eq. 4), `Diffusion.loss` (Algorithm 1 / eq. 14), and `Diffusion.sample`. The sampler has two modes: the literal eq. (11) update, and the eq. (7) posterior-mean form with x0 clipped to [-1, 1] as in the paper's released code. Respacing to K < T steps follows Nichol & Dhariwal section 4. `TinyUNet` is a 0.54M-parameter epsilon network.

## The FID proxy (read this before the numbers)
**This is not FID.** I trained a small CNN on MNIST (97.9% test accuracy) and compute the Frechet distance between its 64-d penultimate features for 500 generated images and for 5,000 real test images. For scale:
- real train images vs the test images, same n = 500: **17.3** (the floor; it reflects finite-sample bias)
- real images + N(0, 0.3^2) pixel noise: **95.0**

## What the ablation found (MNIST, 5,000 training steps per schedule, batch 128, EMA 0.999)
| Sampling steps | linear, clipped-x0 | cosine, clipped-x0 | linear, literal eq. (11) | cosine, literal eq. (11) |
|---|---|---|---|---|
| 1000 | 24.4 | 24.5 | 25.4 | 58.8 |
| 200 | 28.4 | **23.6** | | |
| 50 | 30.1 | 28.5 | 30.0 | **434.8** |
| 20 | 61.7 | **34.6** | | |
| 10 | 138.8 | **64.4** | | |

Figures: `figures/ddpm_ablation.png` (FD-proxy vs steps, abar curves), `figures/ddpm_samples.png`, `figures/forward_process.png`.

## Where the papers and this run disagree, and why
- **At the full 1000 steps the schedule does not matter here** (24.4 vs 24.5). 28x28 MNIST is easy, and the claimed benefit of the cosine schedule was largest on 32x32 and 64x64 natural images.
- **The cosine schedule's real advantage shows up when you cut steps.** At 20 steps it is 34.6 vs 61.7, and at 10 steps 64.4 vs 138.8, about half the distance. Under the linear schedule abar_t is near zero for the last ~40% of t (panel B), so a respaced 10-step chain spends most of its budget in pure-noise territory.
- **Implementation detail that dominated everything:** with the literal eq. (11) update, the cosine model fell apart (58.8 at 1000 steps, 434.8 at 50). Eq. (11) divides by sqrt(alpha_t). The cosine schedule clips beta_T at 0.999, so that first reverse step multiplies any eps error by about 31.6, and with respacing more of the large-beta steps get used. Writing the same update in its eq. (7) posterior form, with x0_hat clipped to the data range (what the paper's released code does), fixes it completely. The paper's equations are correct, but their numerical behaviour differs in float32 on an imperfect network.
- The paper's FID of 3.17 is on CIFAR-10 with a 35.7M-parameter model trained for 800k steps. Nothing here is comparable to it.

## Caveats
One seed, 500 samples per configuration, a proxy metric, and a tiny network trained for 5,000 steps (about 4.5 minutes each). sigma_t^2 = beta_t throughout; the paper reports beta_t and beta~_t give similar results.
