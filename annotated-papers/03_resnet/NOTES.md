# 03 Deep Residual Learning (He et al., 2015) [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

## What the paper claims
- **The degradation problem** (section 1, Fig. 1): as plain nets get deeper, training accuracy saturates and then gets worse. They argue this is not overfitting, because training error rises too, and not vanishing gradients either, because the nets use BN.
- **The fix**, eq. (1): let each block learn a residual, y = F(x, {W_i}) + x. Eq. (2) uses a projection W_s x when the shapes differ. If the identity is already optimal, F can simply go to zero.
- Section 4.2 (CIFAR-10, depth 6n+2): plain-56 ends with higher training error than plain-20, while ResNet-56 beats ResNet-20.

## Where it lives in the code
`model.py`: `Block` (eq. 1, option-A zero-padded identity shortcut as in the CIFAR experiments, plus option B, the eq. 2 projection) and `Net(depth, residual)`. The plain and residual nets share every layer and have identical parameter counts. The only difference is the `+ x`.

## What the ablation found (Fashion-MNIST, 30k training images, 6 epochs, SGD 0.1, wd 1e-4, lr/10 at epochs 3 and 4)
| Depth | Params | Plain train err | Residual train err | Plain test err | Residual test err |
|---|---|---|---|---|---|
| 8 | 75k | 8.05% | 8.10% | 9.88% | 10.35% |
| 20 | 269k | 7.66% | **6.00%** | 9.69% | **8.55%** |
| 32 | 464k | 9.73% | 8.30% | 11.49% | 10.14% |
| 56 | 853k | **17.16%** | 8.68% | **18.27%** | 11.21% |

Train error is measured in eval mode on a fixed 10k-image training subset after training. Gradient at init for depth 56 (`figures/grad_norms_init.png`): in the plain net the gradient on the first block is 916x larger than on the last block. In the residual net that ratio is 4x.

## Where the paper and this run disagree, and why
- **Degradation reproduced.** Plain training error goes 7.7% (20) -> 9.7% (32) -> 17.2% (56). The deeper plain nets have strictly more capacity, yet they fit the training set worse, which is the paper's central observation. With shortcuts, depth 56 stays at 8.7% training error, half the plain-56 figure.
- **Partial disagreement: deeper ResNets did not beat ResNet-20 here.** Residual training error is 6.0 / 8.3 / 8.7% at 20 / 32 / 56. The paper's monotone gains come from 64k iterations, about 160 epochs. In 6 epochs, deeper nets are not yet converged even with shortcuts, since every extra block still has to learn to stay near the identity. So shortcuts remove the degradation, but at this budget they do not make depth pay off.
- At depth 8 the shortcut does nothing (8.05 vs 8.10%). With only 3 blocks, there is nothing to degrade.
- The gradient profile shows a nuance. Plain-56 does not have *vanishing* gradients in the input layers. Its early-layer gradients are huge and wildly unequal across depth (a 916x spread), so a single learning rate is badly mis-scaled for most layers. That is consistent with the paper's statement that BN keeps gradients from vanishing, while optimisation is still hard.

## Caveats
One seed, a 6-epoch compressed schedule, and Fashion-MNIST at 28x28 instead of CIFAR-10 (which was not downloaded to keep compute small). No data augmentation.
