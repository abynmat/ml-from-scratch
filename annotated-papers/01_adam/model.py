"""Adam, RMSProp and SGD+momentum implemented from the paper's update rules.

Paper: Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
       (arXiv:1412.6980). Equation numbers below refer to that paper.
Author: Abyn Scaria.

Each optimiser subclasses torch.optim.Optimizer and implements `step` by hand so
the update rule is visible next to the equation it comes from. These are not
wrappers around torch.optim.Adam; they are the arithmetic written out.
"""
import math

import torch
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    """Adam, Algorithm 1 of Kingma & Ba (2015).

    The exposed knob `bias_correction` lets the ablation turn OFF the
    1/(1 - beta^t) factors and watch what happens, which is the whole point of
    section 3 ("Initialization bias correction").
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 bias_correction=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      bias_correction=bias_correction))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad                                    # g_t
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)             # m_0 <- 0
                    st["v"] = torch.zeros_like(p)             # v_0 <- 0
                st["t"] += 1
                t, m, v = st["t"], st["m"], st["v"]

                # m_t = b1 m_{t-1} + (1 - b1) g_t          (Algorithm 1, line: 1st moment)
                m.mul_(b1).add_(g, alpha=1 - b1)
                # v_t = b2 v_{t-1} + (1 - b2) g_t^2        (Algorithm 1, line: 2nd raw moment)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                if group["bias_correction"]:
                    # m_hat = m_t / (1 - b1^t), v_hat = v_t / (1 - b2^t)   (section 3, eq. 4)
                    m_hat = m / (1 - b1 ** t)
                    v_hat = v / (1 - b2 ** t)
                    # theta_t = theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + eps)
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
                else:
                    # Same step with the raw (biased-toward-zero) moments.
                    p.addcdiv_(m, v.sqrt().add_(eps), value=-lr)
        return closure() if closure is not None else None


class RMSProp(Optimizer):
    """RMSProp (Tieleman & Hinton, 2012), the method Adam reduces to when the
    first-moment average is dropped. Adam's section 5 contrasts the two."""

    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            a, lr, eps = group["alpha"], group["lr"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["v"] = torch.zeros_like(p)
                v = st["v"]
                v.mul_(a).addcmul_(g, g, value=1 - a)         # running mean of g^2
                p.addcdiv_(g, v.sqrt().add_(eps), value=-lr)  # no momentum, no bias correction
        return closure() if closure is not None else None


class SGDMomentum(Optimizer):
    """Classical heavy-ball SGD with momentum, the baseline Adam is pitched against."""

    def __init__(self, params, lr=1e-2, momentum=0.9):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            mu, lr = group["momentum"], group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["b"] = torch.zeros_like(p)
                b = st["b"]
                b.mul_(mu).add_(p.grad)                        # b_t = mu b_{t-1} + g_t
                p.add_(b, alpha=-lr)                           # theta -= lr b_t
        return closure() if closure is not None else None


def build(name, params, lr, bias_correction=True):
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr, bias_correction=bias_correction)
    if name == "adam_nobc":
        return Adam(params, lr=lr, bias_correction=False)
    if name == "rmsprop":
        return RMSProp(params, lr=lr)
    if name == "sgdm":
        return SGDMomentum(params, lr=lr)
    raise ValueError(name)
