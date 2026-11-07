"""
grad.tensor -- an n-dimensional Tensor with reverse-mode automatic differentiation.

Author: Abyn Scaria. MIT licence.

Design in one paragraph
-----------------------
Every operation returns a new Tensor that remembers (a) its parent Tensors and
(b) a *vector-Jacobian product* closure ``vjp(g) -> (g_parent_0, g_parent_1, ...)``.
The closure is pure: it receives the gradient flowing into the output and
*returns* the gradient for each parent instead of mutating the parents. The
engine (``Tensor.backward``) owns all bookkeeping: it builds the tape (a
topological order of the graph), walks it in reverse, sums gradients that
arrive at the same node from several paths, and -- centrally, in exactly one
place -- undoes NumPy broadcasting with :func:`unbroadcast`. Keeping
broadcasting logic out of the individual ops is what keeps each op's backward
rule a one-liner that reads like the maths.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

DEFAULT_DTYPE = np.float32
_GRAD_ENABLED = True

ArrayLike = Union["Tensor", np.ndarray, float, int, Sequence]


@contextlib.contextmanager
def no_grad():
    """Context manager: operations inside do not record a graph (like torch.no_grad)."""
    global _GRAD_ENABLED
    prev, _GRAD_ENABLED = _GRAD_ENABLED, False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


def set_default_dtype(dtype):
    """float32 for speed (default) or float64 for gradient checking."""
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = np.dtype(dtype).type


def get_default_dtype():
    return DEFAULT_DTYPE


def unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` by summing over broadcast dimensions.

    NumPy broadcasting (a) prepends size-1 axes and (b) stretches size-1 axes.
    The adjoint of "copy along an axis" is "sum along that axis", so we sum the
    prepended axes away and sum (keepdims) over every axis that was 1 in the
    original shape but is >1 in the gradient.
    """
    if grad.shape == tuple(shape):
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad.reshape(shape)


def _norm_axes(axis, ndim) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(a % ndim for a in axis)


class Tensor:
    """A NumPy array plus the information needed to differentiate through it."""

    def __init__(self, data: ArrayLike, requires_grad: bool = False, dtype=None,
                 name: str = ""):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, (np.ndarray, np.generic)):   # np.generic: 0-d results of reductions
            arr = np.asarray(data) if dtype is None else np.asarray(data).astype(dtype, copy=False)
            if not np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(DEFAULT_DTYPE)
        else:
            arr = np.asarray(data, dtype=dtype or DEFAULT_DTYPE)
        self.data: np.ndarray = arr
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = bool(requires_grad)
        self.name = name
        # graph bookkeeping (empty for leaves)
        self._parents: Tuple["Tensor", ...] = ()
        self._vjp: Optional[Callable[[np.ndarray], Tuple[Optional[np.ndarray], ...]]] = None
        self._op = ""
        self._retain = False

    # ------------------------------------------------------------------ basics
    shape = property(lambda self: self.data.shape)
    ndim = property(lambda self: self.data.ndim)
    size = property(lambda self: self.data.size)
    dtype = property(lambda self: self.data.dtype)
    is_leaf = property(lambda self: not self._parents)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        tag = f", op={self._op}" if self._op else ""
        rg = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({np.array2string(self.data, precision=4, threshold=8)}{rg}{tag})"

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data.reshape(-1)[0]) if self.data.size == 1 else float(self.data)

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    def retain_grad(self) -> "Tensor":
        """Keep ``.grad`` on this non-leaf node after backward (used by the visualiser)."""
        self._retain = True
        return self

    def zero_grad(self):
        self.grad = None

    # -------------------------------------------------------------- factories
    @staticmethod
    def zeros(*shape, requires_grad=False, dtype=None):
        return Tensor(np.zeros(shape, dtype=dtype or DEFAULT_DTYPE), requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False, dtype=None):
        return Tensor(np.ones(shape, dtype=dtype or DEFAULT_DTYPE), requires_grad)

    @staticmethod
    def randn(*shape, requires_grad=False, dtype=None, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        return Tensor(rng.standard_normal(shape).astype(dtype or DEFAULT_DTYPE), requires_grad)

    # ------------------------------------------------------------ graph core
    def _lift(self, other: ArrayLike) -> "Tensor":
        """Wrap a Python scalar / array as a constant Tensor of *our* dtype."""
        if isinstance(other, Tensor):
            return other
        return Tensor(np.asarray(other, dtype=self.data.dtype))

    @staticmethod
    def _make(data: np.ndarray, parents: Tuple["Tensor", ...], op: str, vjp) -> "Tensor":
        out = Tensor(data)
        if _GRAD_ENABLED and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = parents
            out._vjp = vjp
            out._op = op
        return out

    def tape(self) -> list:
        """Topological order of every node that requires grad and leads to ``self``.

        Iterative DFS (post-order) so very deep graphs -- e.g. the scalar
        reference engine's millions of nodes -- do not hit Python's recursion limit.
        """
        order, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.requires_grad and id(p) not in seen:
                    stack.append((p, False))
        return order

    def backward(self, grad: Optional[ArrayLike] = None, retain_graph: bool = False):
        """Reverse-mode sweep: accumulate d(self)/d(leaf) into every leaf's ``.grad``."""
        if not self.requires_grad:
            raise RuntimeError("backward() on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad.data if isinstance(grad, Tensor) else grad, dtype=self.data.dtype)

        order = self.tape()
        grads = {id(self): grad}           # adjoints of in-flight nodes
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node.is_leaf or node._retain:
                node.grad = g.copy() if node.grad is None else node.grad + g
            if node.is_leaf:
                continue
            parent_grads = node._vjp(g)
            for p, pg in zip(node._parents, parent_grads):
                if pg is None or not p.requires_grad:
                    continue
                if pg.shape != p.data.shape:            # the ONE place broadcasting is undone
                    pg = unbroadcast(pg, p.data.shape)
                k = id(p)
                grads[k] = pg if k not in grads else grads[k] + pg
            if not retain_graph:                        # free the graph as we go
                node._parents, node._vjp = (), None

    # ------------------------------------------------------- elementwise ops
    def __add__(self, other):
        other = self._lift(other)
        return Tensor._make(self.data + other.data, (self, other), "add", lambda g: (g, g))

    __radd__ = __add__

    def __sub__(self, other):
        other = self._lift(other)
        return Tensor._make(self.data - other.data, (self, other), "sub", lambda g: (g, -g))

    def __rsub__(self, other):
        return self._lift(other) - self

    def __mul__(self, other):
        other = self._lift(other)
        a, b = self.data, other.data
        return Tensor._make(a * b, (self, other), "mul", lambda g: (g * b, g * a))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._lift(other)
        a, b = self.data, other.data
        return Tensor._make(a / b, (self, other), "div",
                            lambda g: (g / b, -g * a / (b * b)))

    def __rtruediv__(self, other):
        return self._lift(other) / self

    def __neg__(self):
        return Tensor._make(-self.data, (self,), "neg", lambda g: (-g,))

    def __pow__(self, exponent):
        a = self.data
        if isinstance(exponent, Tensor):           # tensor ** tensor
            b = exponent.data
            out = a ** b
            return Tensor._make(out, (self, exponent), "pow",
                                lambda g: (g * b * a ** (b - 1), g * out * np.log(a)))
        p = float(exponent)                        # tensor ** constant
        return Tensor._make(a ** p, (self,), f"pow{p:g}",
                            lambda g: (g * p * a ** (p - 1),))

    def __rpow__(self, base):
        return self._lift(base) ** self

    def exp(self):
        out = np.exp(self.data)
        return Tensor._make(out, (self,), "exp", lambda g: (g * out,))

    def log(self):
        a = self.data
        return Tensor._make(np.log(a), (self,), "log", lambda g: (g / a,))

    def sqrt(self):
        return self ** 0.5

    def tanh(self):
        out = np.tanh(self.data)
        return Tensor._make(out, (self,), "tanh", lambda g: (g * (1.0 - out * out),))

    def relu(self):
        mask = self.data > 0
        return Tensor._make(self.data * mask, (self,), "relu", lambda g: (g * mask,))

    def sigmoid(self):
        # Stable in both tails: never evaluates exp of a large positive number.
        a = self.data
        e = np.exp(-np.abs(a))
        out = np.where(a >= 0, 1.0 / (1.0 + e), e / (1.0 + e)).astype(a.dtype)
        return Tensor._make(out, (self,), "sigmoid", lambda g: (g * out * (1.0 - out),))

    # --------------------------------------------------------- linear algebra
    def __matmul__(self, other):
        other = self._lift(other)
        A, B = self.data, other.data
        # Treat vectors as 1xK / Kx1 matrices so one backward rule covers all cases.
        A2 = A[None, :] if A.ndim == 1 else A
        B2 = B[:, None] if B.ndim == 1 else B
        # NumPy 2.0 on macOS/Accelerate can raise *spurious* divide/overflow flags inside
        # matmul even when every output is finite (the parity tests confirm the values),
        # so FP warnings are silenced for matmul only. Real overflow still shows up as inf.
        with np.errstate(all="ignore"):
            out2 = A2 @ B2
            out = A @ B

        def vjp(g):
            g2 = g.reshape(out2.shape)
            with np.errstate(all="ignore"):
                gA = g2 @ np.swapaxes(B2, -1, -2)
                gB = np.swapaxes(A2, -1, -2) @ g2
            gA = unbroadcast(gA, A2.shape).reshape(A.shape)
            gB = unbroadcast(gB, B2.shape).reshape(B.shape)
            return gA, gB

        return Tensor._make(out, (self, other), "matmul", vjp)

    def __rmatmul__(self, other):
        return self._lift(other) @ self

    # ------------------------------------------------------------ reductions
    def sum(self, axis=None, keepdims=False):
        shape, axes = self.data.shape, _norm_axes(axis, self.data.ndim)

        def vjp(g):
            if not keepdims:
                g = np.expand_dims(g, axes)
            return (np.broadcast_to(g, shape),)

        return Tensor._make(self.data.sum(axis=axes, keepdims=keepdims), (self,), "sum", vjp)

    def mean(self, axis=None, keepdims=False):
        axes = _norm_axes(axis, self.data.ndim)
        n = int(np.prod([self.data.shape[a] for a in axes])) if axes else 1
        return self.sum(axis=axes, keepdims=keepdims) * (1.0 / n)

    def var(self, axis=None, keepdims=False):
        """Biased (population) variance, as used by LayerNorm."""
        mu = self.mean(axis=axis, keepdims=True)
        return ((self - mu) ** 2).mean(axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """Max reduction. Ties share the gradient equally (same rule as torch.amax)."""
        a, axes = self.data, _norm_axes(axis, self.data.ndim)
        m = a.max(axis=axes, keepdims=True)

        def vjp(g):
            if not keepdims:
                g = np.expand_dims(g, axes)
            mask = (a == m)
            return (mask * g / mask.sum(axis=axes, keepdims=True),)

        out = m if keepdims else m.squeeze(axis=axes)
        return Tensor._make(out, (self,), "max", vjp)

    def logsumexp(self, axis=-1, keepdims=False):
        a, axes = self.data, _norm_axes(axis, self.data.ndim)
        m = a.max(axis=axes, keepdims=True)
        lse = m + np.log(np.exp(a - m).sum(axis=axes, keepdims=True))
        soft = np.exp(a - lse)

        def vjp(g):
            if not keepdims:
                g = np.expand_dims(g, axes)
            return (g * soft,)

        out = lse if keepdims else lse.squeeze(axis=axes)
        return Tensor._make(out, (self,), "logsumexp", vjp)

    def softmax(self, axis=-1):
        a = self.data
        e = np.exp(a - a.max(axis=axis, keepdims=True))   # shift: exp never overflows
        s = e / e.sum(axis=axis, keepdims=True)
        # J^T g for softmax: s * (g - <g, s>)
        return Tensor._make(s, (self,), "softmax",
                            lambda g: (s * (g - (g * s).sum(axis=axis, keepdims=True)),))

    def log_softmax(self, axis=-1):
        a = self.data
        z = a - a.max(axis=axis, keepdims=True)
        out = z - np.log(np.exp(z).sum(axis=axis, keepdims=True))
        s = np.exp(out)
        return Tensor._make(out, (self,), "log_softmax",
                            lambda g: (g - s * g.sum(axis=axis, keepdims=True),))

    # ---------------------------------------------------------- shape ops
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        old = self.data.shape
        return Tensor._make(self.data.reshape(shape), (self,), "reshape",
                            lambda g: (g.reshape(old),))

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        axes = axes or tuple(reversed(range(self.data.ndim)))
        inv = tuple(np.argsort(axes))
        return Tensor._make(self.data.transpose(axes), (self,), "transpose",
                            lambda g: (g.transpose(inv),))

    T = property(lambda self: self.transpose())

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.data.astype(np.int64)
        elif isinstance(idx, tuple):
            idx = tuple(i.data.astype(np.int64) if isinstance(i, Tensor) else i for i in idx)
        a = self.data

        def vjp(g):
            full = np.zeros_like(a)
            np.add.at(full, idx, g)        # add.at (not +=) so repeated indices accumulate
            return (full,)

        return Tensor._make(a[idx], (self,), "getitem", vjp)


def concatenate(tensors: Iterable[Tensor], axis: int = 0) -> Tensor:
    tensors = list(tensors)
    datas = [t.data for t in tensors]
    out = np.concatenate(datas, axis=axis)
    cuts = np.cumsum([d.shape[axis] for d in datas])[:-1]
    return Tensor._make(out, tuple(tensors), "concat",
                        lambda g: tuple(np.split(g, cuts, axis=axis)))


def stack(tensors: Iterable[Tensor], axis: int = 0) -> Tensor:
    tensors = list(tensors)
    expanded = [t.reshape(t.shape[:axis % (t.ndim + 1)] + (1,) + t.shape[axis % (t.ndim + 1):])
                for t in tensors]
    return concatenate(expanded, axis=axis)
