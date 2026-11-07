"""
One table of every differentiable op, shared by the unit tests (tests/test_ops.py)
and the gradient-check report (gradcheck.py). Author: Abyn Scaria.

Each case = (case name, op family, input shapes, input domains, engine fn, torch fn).
Domains: "any" ~ N(0,1); "pos" ~ U(0.5, 2) for log / div / fractional pow;
"away0" ~ +-U(0.2, 1.5) for relu, whose kink at 0 breaks finite differences.
"""
import numpy as np
import torch

import grad.functional as F
from grad import concatenate

IDX_FANCY = np.array([0, 2, 0, 1])          # repeated index -> exercises np.add.at
ROWS, COLS = np.arange(4), np.array([1, 0, 2, 2])


def _amax(x, dim=None, keepdim=False):
    dim = tuple(range(x.ndim)) if dim is None else dim
    return torch.amax(x, dim=dim, keepdim=keepdim)


CASES = [
    # name, op, shapes, domains, engine, torch
    ("add", "add", [(3, 4), (3, 4)], None, lambda a, b: a + b, lambda a, b: a + b),
    ("add_bcast_row", "add", [(3, 4), (4,)], None, lambda a, b: a + b, lambda a, b: a + b),
    ("add_bcast_3d", "add", [(2, 3, 1), (3, 4)], None, lambda a, b: a + b, lambda a, b: a + b),
    ("sub_bcast_col", "sub", [(3, 4), (3, 1)], None, lambda a, b: a - b, lambda a, b: a - b),
    ("rsub_scalar", "sub", [(3, 4)], None, lambda a: 1.5 - a, lambda a: 1.5 - a),
    ("mul", "mul", [(3, 4), (3, 4)], None, lambda a, b: a * b, lambda a, b: a * b),
    ("mul_bcast", "mul", [(5, 1, 3), (4, 3)], None, lambda a, b: a * b, lambda a, b: a * b),
    ("mul_scalar", "mul", [(3, 4)], None, lambda a: a * 2.5, lambda a: a * 2.5),
    ("div", "div", [(3, 4), (3, 4)], ["any", "pos"], lambda a, b: a / b, lambda a, b: a / b),
    ("div_bcast", "div", [(3, 4), (4,)], ["any", "pos"], lambda a, b: a / b, lambda a, b: a / b),
    ("rdiv_scalar", "div", [(3, 4)], ["pos"], lambda a: 2.0 / a, lambda a: 2.0 / a),
    ("pow_int", "pow", [(3, 4)], None, lambda a: a ** 3, lambda a: a ** 3),
    ("pow_frac", "pow", [(3, 4)], ["pos"], lambda a: a ** 0.5, lambda a: a ** 0.5),
    ("pow_tensor_bcast", "pow", [(3, 4), (4,)], ["pos", "any"], lambda a, b: a ** b,
     lambda a, b: a ** b),
    ("neg", "neg", [(3, 4)], None, lambda a: -a, lambda a: -a),
    ("matmul", "matmul", [(3, 4), (4, 5)], None, lambda a, b: a @ b, lambda a, b: a @ b),
    ("matmul_batched", "matmul", [(2, 3, 4), (4, 5)], None, lambda a, b: a @ b,
     lambda a, b: a @ b),
    ("matmul_bcast_batch", "matmul", [(2, 1, 3, 4), (3, 4, 2)], None, lambda a, b: a @ b,
     lambda a, b: a @ b),
    ("matmul_mat_vec", "matmul", [(3, 4), (4,)], None, lambda a, b: a @ b, lambda a, b: a @ b),
    ("matmul_vec_mat", "matmul", [(4,), (4, 5)], None, lambda a, b: a @ b, lambda a, b: a @ b),
    ("sum_all", "sum", [(3, 4)], None, lambda a: a.sum(), lambda a: a.sum()),
    ("sum_axis0", "sum", [(3, 4)], None, lambda a: a.sum(axis=0), lambda a: a.sum(dim=0)),
    ("sum_axes_keepdims", "sum", [(2, 3, 4)], None, lambda a: a.sum(axis=(0, 2), keepdims=True),
     lambda a: a.sum(dim=(0, 2), keepdim=True)),
    ("mean_all", "mean", [(3, 4)], None, lambda a: a.mean(), lambda a: a.mean()),
    ("mean_axis1", "mean", [(2, 3, 4)], None, lambda a: a.mean(axis=1), lambda a: a.mean(dim=1)),
    ("max_all", "max", [(3, 4)], None, lambda a: a.max(), lambda a: _amax(a)),
    ("max_axis1", "max", [(3, 4)], None, lambda a: a.max(axis=1), lambda a: _amax(a, 1)),
    ("max_axis0_keepdims", "max", [(3, 4)], None, lambda a: a.max(axis=0, keepdims=True),
     lambda a: _amax(a, 0, True)),
    ("exp", "exp", [(3, 4)], None, lambda a: a.exp(), lambda a: a.exp()),
    ("log", "log", [(3, 4)], ["pos"], lambda a: a.log(), lambda a: a.log()),
    ("tanh", "tanh", [(3, 4)], None, lambda a: a.tanh(), lambda a: a.tanh()),
    ("relu", "relu", [(3, 4)], ["away0"], lambda a: a.relu(), lambda a: a.relu()),
    ("sigmoid", "sigmoid", [(3, 4)], None, lambda a: a.sigmoid(), lambda a: a.sigmoid()),
    ("softmax_last", "softmax", [(3, 5)], None, lambda a: a.softmax(-1),
     lambda a: a.softmax(-1)),
    ("softmax_axis0", "softmax", [(3, 5)], None, lambda a: a.softmax(0), lambda a: a.softmax(0)),
    ("log_softmax", "log_softmax", [(3, 5)], None, lambda a: a.log_softmax(-1),
     lambda a: a.log_softmax(-1)),
    ("logsumexp", "logsumexp", [(3, 5)], None, lambda a: a.logsumexp(-1),
     lambda a: a.logsumexp(-1)),
    ("reshape", "reshape", [(3, 4)], None, lambda a: a.reshape(2, 6), lambda a: a.reshape(2, 6)),
    ("transpose_perm", "transpose", [(2, 3, 4)], None, lambda a: a.transpose(2, 0, 1),
     lambda a: a.permute(2, 0, 1)),
    ("transpose_T", "transpose", [(3, 4)], None, lambda a: a.T, lambda a: a.T),
    ("getitem_slice", "getitem", [(4, 5)], None, lambda a: a[1:, ::2], lambda a: a[1:, ::2]),
    ("getitem_fancy_repeat", "getitem", [(3, 5)], None, lambda a: a[IDX_FANCY],
     lambda a: a[torch.as_tensor(IDX_FANCY)]),
    ("getitem_row_col_pairs", "getitem", [(4, 3)], None, lambda a: a[ROWS, COLS],
     lambda a: a[torch.as_tensor(ROWS), torch.as_tensor(COLS)]),
    ("concat_axis0", "concatenate", [(2, 3), (4, 3)], None,
     lambda a, b: concatenate([a, b], 0), lambda a, b: torch.cat([a, b], 0)),
    ("concat_axis1", "concatenate", [(2, 3), (2, 1)], None,
     lambda a, b: concatenate([a, b], 1), lambda a, b: torch.cat([a, b], 1)),
    # composites: many primitives chained, compared against torch's fused kernels
    ("layer_norm", "composite", [(4, 6), (6,), (6,)], None,
     lambda x, w, b: F.layer_norm(x, w, b),
     lambda x, w, b: torch.nn.functional.layer_norm(x, (6,), w, b)),
    ("cross_entropy", "composite", [(4, 3)], None,
     lambda x: F.cross_entropy(x, COLS),
     lambda x: torch.nn.functional.cross_entropy(x, torch.as_tensor(COLS))),
    ("mlp_tanh_bcast", "composite", [(5, 3), (3, 4), (4,)], None,
     lambda x, w, b: ((x @ w + b).tanh() ** 2).mean(axis=0),
     lambda x, w, b: ((x @ w + b).tanh() ** 2).mean(dim=0)),
]


def sample_inputs(shapes, domains, rng):
    domains = domains or ["any"] * len(shapes)
    out = []
    for shape, dom in zip(shapes, domains):
        if dom == "pos":
            x = rng.uniform(0.5, 2.0, size=shape)
        elif dom == "away0":
            x = rng.choice([-1.0, 1.0], size=shape) * rng.uniform(0.2, 1.5, size=shape)
        else:
            x = rng.standard_normal(shape)
        out.append(np.asarray(x, dtype=np.float64))
    return out
