"""
gradcheck.py -- verify analytic gradients (Author: Abyn Scaria).

Library:  ``gradcheck(fn, arrays)`` compares grad's reverse-mode gradient with a
central finite difference  (f(x+h) - f(x-h)) / 2h  computed in float64.

CLI:      ``python3 gradcheck.py`` runs every case in tests/op_specs.py against
both PyTorch autograd and finite differences on 3 random seeds and writes
results/gradcheck.csv plus figures/gradcheck_heatmap.png.

How a vector-valued op is checked: we contract its output with a fixed random
tensor R and differentiate the scalar  L = sum(op(x) * R).  dL/dx = J^T R is
exactly what one backward pass produces, and a random R exercises every
column of the Jacobian with probability 1.

Error metrics (per input, then max over inputs and seeds):
  max_abs = max_i |g_i - ref_i|
  max_rel = max_abs / max(max_i |ref_i|, 1e-12)        (norm-wise relative error)
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from grad import Tensor, no_grad  # noqa: E402


def engine_grads(fn, arrays, R):
    ts = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = fn(*ts)
    (out * Tensor(R)).sum().backward()
    return out.data, [t.grad if t.grad is not None else np.zeros_like(t.data) for t in ts]


def fd_grads(fn, arrays, R, h=1e-6):
    """Central differences, one coordinate at a time, in the arrays' dtype."""
    grads = []
    with no_grad():
        f = lambda xs: float((fn(*[Tensor(x) for x in xs]).data * R).sum())  # noqa: E731
        for k, a in enumerate(arrays):
            g = np.zeros_like(a)
            for i in np.ndindex(a.shape):
                xs = [x.copy() for x in arrays]
                xs[k][i] = a[i] + h
                fp = f(xs)
                xs[k][i] = a[i] - h
                fm = f(xs)
                g[i] = (fp - fm) / (2 * h)
            grads.append(g)
    return grads


def _err(g, ref):
    abs_err = float(np.max(np.abs(g - ref))) if g.size else 0.0
    return abs_err, abs_err / max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1e-12)


def gradcheck(fn, arrays, h=1e-6, seed=0):
    """Return (max_abs, max_rel) of reverse-mode vs finite-difference gradients."""
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    with no_grad():
        out = fn(*[Tensor(a) for a in arrays]).data
    R = np.random.default_rng(seed).standard_normal(out.shape)
    _, g = engine_grads(fn, arrays, R)
    num = fd_grads(fn, arrays, R, h)
    errs = [_err(a, b) for a, b in zip(g, num)]
    return max(e[0] for e in errs), max(e[1] for e in errs)


def torch_grads(fn, arrays, R):
    import torch
    ts = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in arrays]
    out = fn(*ts)
    (out * torch.as_tensor(R)).sum().backward()
    return out.detach().numpy(), [t.grad.numpy() for t in ts]


def run_case(case, seed):
    from tests.op_specs import sample_inputs
    name, op, shapes, domains, f_eng, f_torch = case
    rng = np.random.default_rng(seed)
    arrays = sample_inputs(shapes, domains, rng)
    with no_grad():
        out_shape = f_eng(*[Tensor(a) for a in arrays]).shape
    R = rng.standard_normal(out_shape)
    out_e, g_e = engine_grads(f_eng, arrays, R)
    out_t, g_t = torch_grads(f_torch, arrays, R)
    g_fd = fd_grads(f_eng, arrays, R)
    fwd = float(np.max(np.abs(out_e - out_t)))
    vs_t = [_err(a, b) for a, b in zip(g_e, g_t)]
    vs_fd = [_err(a, b) for a, b in zip(g_e, g_fd)]
    return dict(fwd_abs_vs_torch=fwd,
                grad_abs_vs_torch=max(e[0] for e in vs_t),
                grad_rel_vs_torch=max(e[1] for e in vs_t),
                grad_abs_vs_fd=max(e[0] for e in vs_fd),
                grad_rel_vs_fd=max(e[1] for e in vs_fd))


def main(seeds=(0, 1, 2)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tests.op_specs import CASES

    rows = []
    for case in CASES:
        res = [run_case(case, s) for s in seeds]
        row = {"case": case[0], "op": case[1],
               "input_shapes": " ; ".join("x".join(map(str, s)) for s in case[2])}
        for k in res[0]:
            row[k] = max(r[k] for r in res)
        rows.append(row)
        print(f"{row['case']:<24} vs torch {row['grad_rel_vs_torch']:.1e}   "
              f"vs FD {row['grad_rel_vs_fd']:.1e}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    path = os.path.join(ROOT, "results", "gradcheck.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = ["fwd_abs_vs_torch", "grad_abs_vs_torch", "grad_rel_vs_torch",
            "grad_abs_vs_fd", "grad_rel_vs_fd"]
    M = np.array([[max(r[c], 1e-18) for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 13))
    im = ax.imshow(np.log10(M), aspect="auto", cmap="viridis", vmin=-18, vmax=-6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["case"] for r in rows], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{np.log10(M[i, j]):.0f}", ha="center", va="center",
                    fontsize=6, color="w" if np.log10(M[i, j]) < -12 else "k")
    fig.colorbar(im, ax=ax, label="log10 error (0 shown as -18)")
    ax.set_title("Gradient check, float64, max over 3 seeds\n"
                 "(vs PyTorch: ~1e-16 = bit-level; vs finite diff: ~1e-10 = FD noise)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "gradcheck_heatmap.png"), dpi=150)
    print("wrote", path)
    worst_t = max(r["grad_rel_vs_torch"] for r in rows)
    worst_fd = max(r["grad_rel_vs_fd"] for r in rows)
    print(f"{len(rows)} cases | worst rel err vs torch {worst_t:.2e} | vs FD {worst_fd:.2e}")
    return rows


if __name__ == "__main__":
    main()
