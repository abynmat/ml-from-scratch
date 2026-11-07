"""
The README's worked example, run for real and drawn (Author: Abyn Scaria).

    z = x @ W + b      x: 1x2, W: 2x2, b: 2  (b is broadcast over the batch row)
    h = tanh(z)
    L = mean(h * h)

Prints every forward value and adjoint, checks them against hand-derived
formulas and PyTorch, and writes figures/computation_graph.png and
results/worked_example.json.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

from grad import Tensor  # noqa: E402
from grad.viz import draw  # noqa: E402

x0 = np.array([[1.0, -2.0]])
W0 = np.array([[0.5, -1.0], [0.25, 0.75]])
b0 = np.array([0.1, -0.2])

x = Tensor(x0, requires_grad=True, name="x")
W = Tensor(W0, requires_grad=True, name="W")
b = Tensor(b0, requires_grad=True, name="b")
xW = (x @ W).retain_grad()
xW.name = "xW"
z = (xW + b).retain_grad()
z.name = "z"
h = z.tanh().retain_grad()
h.name = "h"
hh = (h * h).retain_grad()
hh.name = "h2"
L = hh.mean()
L.name = "L"
L.backward(retain_graph=True)

# hand-derived adjoints, written out the way the README derives them
h_v = np.tanh(x0 @ W0 + b0)
dh = 2 * h_v / h_v.size            # dL/dh  = 2h / n
dz = dh * (1 - h_v ** 2)           # dL/dz  = dL/dh * (1 - h^2)
db = dz.sum(0)                     # broadcast in forward -> sum in backward
dW = x0.T @ dz                     # dL/dW  = x^T dz
dx = dz @ W0.T                     # dL/dx  = dz W^T
assert np.allclose(z.grad, dz) and np.allclose(b.grad, db)
assert np.allclose(W.grad, dW) and np.allclose(x.grad, dx)

tx, tW, tb = (torch.tensor(a, requires_grad=True) for a in (x0, W0, b0))
(torch.tanh(tx @ tW + tb) ** 2).mean().backward()
assert np.allclose(tW.grad.numpy(), W.grad) and np.allclose(tx.grad.numpy(), x.grad)

out = {"z": z.data.tolist(), "h": h.data.tolist(), "L": L.item(),
       "dL_dh": h.grad.tolist(), "dL_dz": z.grad.tolist(), "dL_db": b.grad.tolist(),
       "dL_dW": W.grad.tolist(), "dL_dx": x.grad.tolist(),
       "tape_order": [t.name or t._op for t in L.tape()]}
json.dump(out, open(os.path.join(ROOT, "results", "worked_example.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k:>10}: {np.round(v, 4).tolist() if k != 'tape_order' else v}")

draw(L, os.path.join(ROOT, "figures", "computation_graph.png"),
     title="L = mean(tanh(x @ W + b)^2)   forward: left to right, backward: right to left")
print("hand derivation == grad == PyTorch: OK")
