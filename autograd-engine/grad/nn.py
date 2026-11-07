"""
grad.nn -- a small module library on top of grad.Tensor (Author: Abyn Scaria).

Conventions that differ from PyTorch, on purpose:
* ``Linear.weight`` has shape (in_features, out_features) so forward is ``x @ W + b``
  with no transpose. (The parity tests copy ``W.T`` into torch.)
* Every module that needs randomness takes an explicit ``np.random.Generator``;
  there is no hidden global RNG, which makes experiments reproducible.
"""
from __future__ import annotations

import math
from typing import Iterator, List, Optional, Tuple

import numpy as np

from . import functional as F
from .tensor import Tensor, get_default_dtype


class Parameter(Tensor):
    """A Tensor that a Module registers as trainable (stored in the default dtype)."""

    def __init__(self, data, name: str = ""):
        super().__init__(np.asarray(data, dtype=get_default_dtype()), requires_grad=True, name=name)


# ------------------------------------------------------------------ init
class init:
    """Weight-initialisation schemes. Each returns a NumPy array of ``shape``.

    fan_in / fan_out are read from a (in, out) weight shape.
    """

    @staticmethod
    def _fans(shape) -> Tuple[int, int]:
        return (shape[0], shape[1]) if len(shape) == 2 else (shape[-1], shape[-1])

    @staticmethod
    def uniform_fan_in(shape, rng):
        """U(-1/sqrt(fan_in), 1/sqrt(fan_in)); what torch.nn.Linear uses by default."""
        fan_in, _ = init._fans(shape)
        b = 1.0 / math.sqrt(fan_in)
        return rng.uniform(-b, b, size=shape)

    @staticmethod
    def xavier_uniform(shape, rng, gain: float = 1.0):
        """Glorot & Bengio (2010): keeps activation variance ~constant for tanh."""
        fan_in, fan_out = init._fans(shape)
        b = gain * math.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-b, b, size=shape)

    @staticmethod
    def kaiming_normal(shape, rng, gain: float = math.sqrt(2.0)):
        """He et al. (2015): std = gain / sqrt(fan_in); gain sqrt(2) for ReLU."""
        fan_in, _ = init._fans(shape)
        return rng.standard_normal(shape) * gain / math.sqrt(fan_in)

    @staticmethod
    def lecun_normal(shape, rng):
        return init.kaiming_normal(shape, rng, gain=1.0)

    @staticmethod
    def normal(shape, rng, std: float = 1.0):
        return rng.standard_normal(shape) * std

    SCHEMES = ("uniform_fan_in", "xavier_uniform", "kaiming_normal", "lecun_normal")


# ------------------------------------------------------------------ Module
class Module:
    training: bool = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        for key, val in vars(self).items():
            items = [(f"{key}.{i}", v) for i, v in enumerate(val)] if isinstance(val, (list, tuple)) \
                else [(key, val)]
            for name, v in items:
                if isinstance(v, Parameter):
                    yield prefix + name, v
                elif isinstance(v, Module):
                    yield from v.named_parameters(prefix + name + ".")

    def parameters(self) -> List[Parameter]:
        return [p for _, p in self.named_parameters()]

    def modules(self) -> Iterator["Module"]:
        yield self
        for val in vars(self).values():
            for v in (val if isinstance(val, (list, tuple)) else [val]):
                if isinstance(v, Module):
                    yield from v.modules()

    def train(self, mode: bool = True) -> "Module":
        for m in self.modules():
            m.training = mode
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init_scheme: str = "uniform_fan_in", rng: Optional[np.random.Generator] = None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.weight = Parameter(getattr(init, init_scheme)((in_features, out_features), rng), "W")
        if bias:
            if init_scheme == "uniform_fan_in":
                b = 1.0 / math.sqrt(in_features)
                self.bias = Parameter(rng.uniform(-b, b, size=out_features), "b")
            else:
                self.bias = Parameter(np.zeros(out_features), "b")
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = Parameter(np.ones(dim), "gamma")
        self.bias = Parameter(np.zeros(dim), "beta")
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight, self.bias, self.eps)


class Dropout(Module):
    def __init__(self, p: float = 0.5, rng: Optional[np.random.Generator] = None):
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training, self.rng)


class Embedding(Module):
    def __init__(self, num_embeddings: int, dim: int, rng: Optional[np.random.Generator] = None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.weight = Parameter(rng.standard_normal((num_embeddings, dim)), "E")

    def forward(self, idx) -> Tensor:
        return self.weight[np.asarray(idx, dtype=np.int64)]


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MSELoss(Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class CrossEntropyLoss(Module):
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets)
