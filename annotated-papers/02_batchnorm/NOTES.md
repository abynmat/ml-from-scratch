# 02 Batch Normalization (Ioffe & Szegedy, 2015) vs Layer Normalization (Ba et al., 2016)

[BN arXiv:1502.03167](https://arxiv.org/abs/1502.03167) · [LN arXiv:1607.06450](https://arxiv.org/abs/1607.06450)

## What the papers claim
- BN, Algorithm 1: normalise each activation with mini-batch mean and variance, then apply a learned gamma, beta. Algorithm 2: at inference, swap the batch statistics for population estimates, using the unbiased variance m/(m-1).
- BN abstract: BN allows "much higher learning rates" and less careful initialisation, and reaches the same ImageNet accuracy with 14x fewer steps.
- LN, eq. (3): compute the statistics over all hidden units of a single example instead, so training and test behave identically. Section 6.7 reports that LN speeds up ConvNets over no normalisation, but BN still wins on ConvNets.

## Where it lives in the code
`model.py`: `BatchNorm2d` (Algorithm 1 in training, Algorithm 2 running statistics at eval) and `LayerNorm2d` (eq. 3 and eq. 5). A 6-conv CNN can use either one or no normalisation.

## What the ablation found (Fashion-MNIST, SGD momentum 0.9, batch 128, 2 epochs)
| | lr 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|---|
| no norm | 0.140 | 0.717 | 0.805 | diverged | diverged | 0.100 (chance) |
| **BatchNorm** | 0.846 | 0.877 | 0.865 | 0.892 | **0.896** | 0.873 |
| LayerNorm | 0.841 | 0.839 | 0.840 | 0.835 | 0.774 | 0.783 |

- Parity with `torch.nn.BatchNorm2d` (train output, running stats, eval output) and `F.layer_norm`: all within 1e-6.
- Usable LR range: without normalisation, only 0.01-0.03 trains at all. BN reaches at least 0.85 at every LR from 0.01 to 1.0 (0.846 at 0.003), a span of more than two orders of magnitude.
- Speed at lr = 0.03: BN hits 85% on the 2k test subset after 400 steps, LN after 900, and no-norm never does within 2 epochs (best 80.5%).

### The train/eval-mode bug (same trained BN weights, lr 0.1)
| How the test set was evaluated | Test acc |
|---|---|
| eval mode (correct) | 0.892 |
| train mode, shuffled batches of 500 | 0.908 |
| train mode, batches of 16 | 0.890 |
| train mode, batches of 2 | **0.637** |
| train mode, batches of 500 sorted by class | **0.152** |
| eval mode, running stats left at init (0 / 1) | **0.154** |

## Where the paper and this run disagree, and why
- **"Higher learning rates" holds clearly.** The paper's mechanism, reducing "internal covariate shift", is a separate question. Santurkar et al. ([arXiv:1805.11604](https://arxiv.org/abs/1805.11604)) argue the benefit comes from a smoother loss landscape instead. This ablation measures the effect, not the mechanism.
- **LN on a ConvNet trailed BN by 0.5-12 points (more at high LR)**, which is what the LN paper itself reports in section 6.7. Their explanation is that units near the image border have very different statistics from the rest of the layer, so normalising them all together is a poor fit.
- **Forgetting `model.eval()` is not always a disaster, and that is what makes it dangerous.** With large shuffled batches, per-batch statistics are a good estimate, so accuracy stays the same or even rises by 1.6 points. The rise is transductive: the net is borrowing statistics from the test batch. My guess is that the EMA running estimate also lags behind the weights after 2 epochs at a constant lr of 0.1. That part is a hypothesis I did not test. The bug only shows up when batches are tiny, or when a batch is not i.i.d. (sorted by class gives 15%). Both happen easily in production inference.
- The init-stats row shows that buffers matter as much as weights. Loading a checkpoint without its `running_mean`/`running_var` produces the same collapse.

## Caveats
One seed and 2 epochs. The "no norm" net uses PyTorch's default init with conv biases, and better init (e.g. Fixup) narrows the gap. LR = 1.0 without normalisation did not trip the divergence check (loss stayed finite) but accuracy stayed at chance; I did not inspect why (dead ReLUs are the likely cause, unverified).
