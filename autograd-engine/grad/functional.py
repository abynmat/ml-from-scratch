"""
grad.functional -- stateless functions built from Tensor ops (Author: Abyn Scaria).

Everything here is a composition of primitives in ``grad.tensor``, so none of
these functions needs its own backward rule. The two ``naive_*`` functions are
deliberately unstable and exist only for the numerical-stability study.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .tensor import Tensor, concatenate, stack  # noqa: F401  (re-exported)


def relu(x: Tensor) -> Tensor:
    return x.relu()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.softmax(axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.log_softmax(axis)


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """x @ W + b, with W stored as (in_features, out_features)."""
    y = x @ weight
    return y + bias if bias is not None else y


def layer_norm(x: Tensor, weight: Optional[Tensor], bias: Optional[Tensor],
               eps: float = 1e-5) -> Tensor:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = (x - mu) / (var + eps) ** 0.5
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def dropout(x: Tensor, p: float, training: bool, rng: np.random.Generator) -> Tensor:
    """Inverted dropout: scale kept units by 1/(1-p) so eval needs no rescaling."""
    if not training or p == 0.0:
        return x
    keep = (rng.random(x.shape) >= p).astype(x.dtype) / (1.0 - p)
    return x * Tensor(keep)


def mse_loss(pred: Tensor, target) -> Tensor:
    diff = pred - target
    return (diff * diff).mean()


def cross_entropy(logits: Tensor, targets) -> Tensor:
    """Mean negative log-likelihood of integer ``targets`` under softmax(logits).

    Uses log_softmax, i.e. the log-sum-exp trick, so it is finite for any logit scale.
    """
    targets = np.asarray(targets.data if isinstance(targets, Tensor) else targets, dtype=np.int64)
    logp = logits.log_softmax(axis=-1)
    picked = logp[np.arange(len(targets)), targets]
    return -picked.mean()


# ----------------------------------------------- deliberately naive versions
def naive_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """exp(x) / sum exp(x) with NO max-shift: overflows once a logit exceeds ~88 (float32)."""
    e = x.exp()
    return e / e.sum(axis=axis, keepdims=True)


def naive_cross_entropy(logits: Tensor, targets) -> Tensor:
    """-log(naive_softmax[target]): overflows (NaN) for large logits and gives
    log(0) = -inf when the target probability underflows."""
    targets = np.asarray(targets, dtype=np.int64)
    p = naive_softmax(logits, axis=-1)
    return -(p[np.arange(len(targets)), targets].log()).mean()
