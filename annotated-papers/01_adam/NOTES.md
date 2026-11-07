# 01 Adam (Kingma & Ba, 2014) [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

## What the paper claims
- Algorithm 1: keep exponential moving averages of the gradient (m_t) and the squared gradient (v_t), divide both by (1 - beta^t), and step by alpha * m_hat / (sqrt(v_hat) + eps).
- Section 3, eq. (1)-(4): because m_0 = v_0 = 0, E[v_t] = E[g^2] (1 - beta2^t) + zeta. So the raw averages are biased toward zero early in training, and dividing by (1 - beta2^t) removes that bias. The paper says skipping the correction gives much larger initial steps. Their section 6.4 VAE experiment reports instability without it when beta2 is close to 1.
- Section 6: Adam matches or beats SGD with momentum and RMSProp-style methods on MLPs, CNNs and logistic regression.

## Where it lives in the code
`model.py`, class `Adam`. Each line of the update carries its Algorithm 1 line or its eq. number. `bias_correction=False` keeps the raw m_t / sqrt(v_t). `RMSProp` and `SGDMomentum` are written out by hand in the same file.

## What the ablation found (MNIST MLP 784-256-256-10, 3 epochs; noisy quadratic, d=100)
| Test | Result |
|---|---|
| Parity with `torch.optim.Adam`, 200 steps | max abs parameter difference 9.1e-07 (float32 round-off) |
| Peak step-size ratio, no-correction / correction | 5.2x measured at t=5. Closed form (1-b1^t)/sqrt(1-b2^t) peaks at 6.6x near t=12 |
| MNIST, best LR per optimiser (test acc) | Adam 0.972, Adam without correction **0.978**, RMSProp 0.964, SGD+momentum 0.971 |
| Noisy quadratic, mean f over last 300 steps | Adam 0.094, no correction 0.085, RMSProp 0.093, SGD+momentum 0.436 |
| Stress test: beta2 = 0.9999, lr = 1e-2, 1 epoch | **with correction 0.958, without 0.110 (loss spiked to 232, never recovered)** |
| Stress test: beta2 = 0.999, lr = 1e-2 | with 0.955, without 0.875 |
| beta2 = 0.99, any LR | within 0.5 points of each other |

Exact numbers are in `results/results.json`. Figures: `figures/adam_ablation.png` and `figures/bias_correction_stress.png`.

## Where the paper and this run disagree, and why
- **At default settings, bias correction did not help here.** Without it, Adam takes up to about 5x larger steps for the first few hundred iterations. On a small, well-conditioned MLP with a 3-epoch budget, that works like a free LR boost, so the uncorrected run comes out slightly ahead. This matches recent re-examinations of the correction (e.g. [arXiv:2511.20516](https://arxiv.org/abs/2511.20516)).
- **The paper's actual claim holds where it says it should.** The penalty for skipping correction grows with beta2 because the early-step blow-up is about 1/sqrt(1 - beta2^t). At beta2 = 0.9999 and lr = 1e-2, the first steps are roughly 10-30x too large and the network dies (11% accuracy). The corrected run trains normally. So bias correction works as insurance for large beta2 and LR, not as a speed-up.
- The measured ratio falls below the closed form after about t=5. The closed form assumes stationary gradients (zeta = 0 in eq. 4). In real training the gradient shrinks quickly early on, so the old, larger g^2 terms in v_t keep the uncorrected denominator larger than the formula predicts.
- On the ill-conditioned quadratic, all three adaptive methods beat SGD+momentum by about 5x. That matches the per-coordinate-scaling argument in section 2.1. SGD's best LR there is small because its noise floor scales with lr * h_max.

## Caveats
One seed, a small MLP, and a 3-point LR grid per optimiser. SGD+momentum's MNIST optimum sits in the middle of its grid (0.1). Every other optimiser's best LR is also inside its grid.
