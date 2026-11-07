"""
grad.scalar -- a ~50-line *scalar* reverse-mode engine, used only as a baseline
for the timing study (Author: Abyn Scaria).

It is the "one graph node per number" design popularised by micrograd
(https://github.com/karpathy/micrograd), written from scratch here so the
performance comparison is apples-to-apples with the same Python interpreter.
"""
import math
import random


class Scalar:
    __slots__ = ("v", "g", "parents", "local")

    def __init__(self, v, parents=(), local=()):
        self.v, self.g = v, 0.0
        self.parents, self.local = parents, local   # local[i] = d self / d parents[i]

    def __add__(self, o):
        o = o if isinstance(o, Scalar) else Scalar(o)
        return Scalar(self.v + o.v, (self, o), (1.0, 1.0))

    def __mul__(self, o):
        o = o if isinstance(o, Scalar) else Scalar(o)
        return Scalar(self.v * o.v, (self, o), (o.v, self.v))

    __radd__, __rmul__ = __add__, __mul__

    def tanh(self):
        t = math.tanh(self.v)
        return Scalar(t, (self,), (1.0 - t * t,))

    def backward(self):
        order, seen, stack = [], set(), [(self, False)]
        while stack:                                   # iterative post-order DFS
            n, done = stack.pop()
            if done:
                order.append(n)
            elif id(n) not in seen:
                seen.add(id(n))
                stack.append((n, True))
                stack.extend((p, False) for p in n.parents)
        self.g = 1.0
        for n in reversed(order):
            for p, d in zip(n.parents, n.local):
                p.g += d * n.g


def scalar_mlp(sizes, seed=0):
    """List of layers; each layer is a list of (weights, bias) neurons."""
    rnd = random.Random(seed)
    return [[([Scalar(rnd.uniform(-1, 1) / math.sqrt(i)) for _ in range(i)], Scalar(0.0))
             for _ in range(o)] for i, o in zip(sizes[:-1], sizes[1:])]


def scalar_forward(layers, x):
    for li, layer in enumerate(layers):
        x = [sum((w * xi for w, xi in zip(ws, x)), b) for ws, b in layer]
        if li < len(layers) - 1:
            x = [xi.tanh() for xi in x]
    return x
