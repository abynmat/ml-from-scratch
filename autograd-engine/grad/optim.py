"""
grad.optim -- SGD (+momentum / Nesterov), Adam and AdamW (Author: Abyn Scaria).

Update rules follow the equations in each paper and the documented PyTorch
semantics (so tests/test_nn.py can check parity step by step):
* SGD momentum: v <- mu*v + g ; p <- p - lr*v         (v starts at g on step 1)
* Adam (Kingma & Ba, 2015) with bias correction; ``weight_decay`` is L2 added to g.
* AdamW (Loshchilov & Hutter, 2019): weight decay is applied to p directly,
  decoupled from the adaptive step.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float):
        self.params: List[Tensor] = list(params)
        self.lr = lr
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self._update(i, p, p.grad)

    def _update(self, i, p, g):  # pragma: no cover
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr)
        self.momentum, self.weight_decay, self.nesterov = momentum, weight_decay, nesterov
        self.buf = [None] * len(self.params)

    def _update(self, i, p, g):
        if self.weight_decay:
            g = g + self.weight_decay * p.data
        if self.momentum:
            self.buf[i] = g.copy() if self.buf[i] is None else self.momentum * self.buf[i] + g
            g = g + self.momentum * self.buf[i] if self.nesterov else self.buf[i]
        p.data -= (self.lr * g).astype(p.data.dtype)


class Adam(Optimizer):
    decoupled = False

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def _update(self, i, p, g):
        if self.weight_decay:
            if self.decoupled:
                p.data *= (1.0 - self.lr * self.weight_decay)
            else:
                g = g + self.weight_decay * p.data
        self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
        self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
        m_hat = self.m[i] / (1 - self.b1 ** self.t)
        v_hat = self.v[i] / (1 - self.b2 ** self.t)
        p.data -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(p.data.dtype)


class AdamW(Adam):
    decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr, betas, eps, weight_decay)
