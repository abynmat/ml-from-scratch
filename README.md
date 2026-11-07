# ml-from-scratch

Machine learning re-implemented from the equations, then checked against the reference library on the same data.
The repo has three parts: classic algorithms in NumPy, a tensor autograd engine with a neural-network library
on top, and six landmark deep-learning papers each paired with an ablation that tests its central claim.
Every table and figure comes from code run in this repository.

| Folder | What it is | Headline result |
|---|---|---|
| [`classic-ml/`](classic-ml/) | 20 estimators in about 1,560 lines of NumPy behind one fit/predict API: linear, ridge, lasso, logistic, kNN, naive Bayes, CART, random forest, gradient boosting, SMO SVMs, k-means++, GMM, PCA, MLP. Parity benchmark, failure-mode study, timing-vs-theory study, 30 tests. | Nine estimators match scikit-learn exactly (agreement 1.000). The RBF SVM finds the same 104 support vectors on breast_cancer. Six failure-mode experiments show where the simple versions break. |
| [`autograd-engine/`](autograd-engine/) | Reverse-mode automatic differentiation on tensors with broadcasting, a tape, `nn`/`optim` modules, a computation-graph visualiser, and a numerical-stability study. | 48 op gradient checks: worst relative error 5.4e-16 vs PyTorch, 1.3e-9 vs finite differences. Trained from identical weights and batches, the engine tracks PyTorch's loss within 6.7e-8 for 30 epochs. Naive cross-entropy overflows at logit scale 23.7 in float32; log-sum-exp holds to 1e4. |
| [`annotated-papers/`](annotated-papers/) | Adam, BatchNorm, ResNet, VAE, GAN and DDPM, each in one readable file with the paper's equations cited by number, a `NOTES.md` on what the paper claims vs what the ablation found, and a shared training harness. | At 1,000 sampling steps the linear and cosine DDPM schedules tie; at 20 steps cosine is clearly better (FD-proxy 34.6 vs 61.7). The literal eq. (11) sampler fails without the clipped-x0 step from the released code. |

## What's different from the well-known from-scratch repos

The popular collections (ML-From-Scratch, homemade-machine-learning, numpy-ml, micrograd, the annotated paper
implementations) inspired the scope. None of them ship a parity table against the reference library, a
failure-mode study, a gradient check against PyTorch, or an ablation per paper. Those are the point of this repo.
Each subfolder's README has a "What's different here" section and credits the inspirations.

## Reading order

1. `classic-ml` for the algorithms most interviews and most tabular work rely on.
2. `autograd-engine` for how a deep-learning framework computes gradients.
3. `annotated-papers` for the ideas that made deep learning train.

## Run it

Each folder has its own `README.md` with exact commands and a `requirements.txt`. Tests run with plain `python3`.
Datasets come from scikit-learn loaders or a `data/download.py` script; nothing large is committed.

## License

MIT, see [LICENSE](LICENSE).
