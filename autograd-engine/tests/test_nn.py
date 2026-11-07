"""
Module / loss / optimiser parity with PyTorch (Author: Abyn Scaria).
Run as plain Python:  python3 tests/test_nn.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

import grad  # noqa: E402
from grad import Tensor, nn, optim  # noqa: E402
import grad.functional as F  # noqa: E402
from tests.test_ops import run  # noqa: E402

grad.set_default_dtype(np.float64)
torch.set_default_dtype(torch.float64)
RNG = np.random.default_rng(0)


def _mlp_pair():
    """Same 3-layer MLP in grad and torch, weights copied across."""
    ours = nn.Sequential(nn.Linear(4, 8, rng=RNG), nn.Tanh(), nn.Linear(8, 8, rng=RNG),
                         nn.ReLU(), nn.LayerNorm(8), nn.Linear(8, 3, rng=RNG))
    theirs = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8),
                                 torch.nn.ReLU(), torch.nn.LayerNorm(8), torch.nn.Linear(8, 3))
    with torch.no_grad():
        for a, b in zip(ours.layers, theirs):
            if isinstance(a, nn.Linear):
                b.weight.copy_(torch.as_tensor(a.weight.data.T))
                b.bias.copy_(torch.as_tensor(a.bias.data))
    return ours, theirs


def _torch_grads_in_our_layout(theirs):
    out = []
    for m in theirs:
        if isinstance(m, torch.nn.Linear):
            out += [m.weight.grad.numpy().T, m.bias.grad.numpy()]
        elif isinstance(m, torch.nn.LayerNorm):
            out += [m.weight.grad.numpy(), m.bias.grad.numpy()]
    return out


def test_mlp_cross_entropy_grads_match_torch():
    ours, theirs = _mlp_pair()
    x = RNG.standard_normal((16, 4))
    y = RNG.integers(0, 3, 16)
    loss = F.cross_entropy(ours(Tensor(x)), y)
    loss.backward()
    tl = torch.nn.functional.cross_entropy(theirs(torch.as_tensor(x)), torch.as_tensor(y))
    tl.backward()
    assert abs(loss.item() - tl.item()) < 1e-12
    for g, tg in zip([p.grad for p in ours.parameters()], _torch_grads_in_our_layout(theirs)):
        assert np.max(np.abs(g - tg)) < 1e-12


def test_mse_matches_torch():
    a, b = RNG.standard_normal((5, 3)), RNG.standard_normal((5, 3))
    ta = torch.tensor(a, requires_grad=True)
    tl = torch.nn.functional.mse_loss(ta, torch.as_tensor(b))
    tl.backward()
    x = Tensor(a, requires_grad=True)
    loss = F.mse_loss(x, b)
    loss.backward()
    assert abs(loss.item() - tl.item()) < 1e-14 and np.allclose(x.grad, ta.grad.numpy())


def test_embedding_grad_accumulates_repeated_ids():
    emb = nn.Embedding(6, 4, rng=RNG)
    temb = torch.nn.Embedding(6, 4)
    with torch.no_grad():
        temb.weight.copy_(torch.as_tensor(emb.weight.data))
    ids = np.array([[1, 2, 1], [5, 1, 0]])
    w = RNG.standard_normal((2, 3, 4))
    (emb(ids) * Tensor(w)).sum().backward()
    (temb(torch.as_tensor(ids)) * torch.as_tensor(w)).sum().backward()
    assert np.allclose(emb.weight.grad, temb.weight.grad.numpy(), atol=1e-14)


def test_dropout_train_eval():
    d = nn.Dropout(0.3, rng=np.random.default_rng(1))
    x = Tensor(np.ones((2000, 50)))
    y = d(x).data
    assert abs((y == 0).mean() - 0.3) < 0.01 and abs(y.mean() - 1.0) < 0.02
    d.eval()
    assert np.array_equal(d(x).data, x.data)


def test_parameter_discovery():
    ours, _ = _mlp_pair()
    names = [n for n, _ in ours.named_parameters()]
    assert len(names) == 8 and names[0] == "layers.0.weight", names
    assert ours.num_parameters() == 4 * 8 + 8 + 8 * 8 + 8 + 8 + 8 + 8 * 3 + 3


def _optimizer_parity(make_ours, make_theirs, steps=25):
    p0 = RNG.standard_normal((3, 4))
    A = RNG.standard_normal((4, 4))
    ours_p = nn.Parameter(p0.copy())
    t_p = torch.tensor(p0.copy(), requires_grad=True)
    o, t = make_ours([ours_p]), make_theirs([t_p])
    for _ in range(steps):   # loss = sum(tanh(p @ A)^2): non-quadratic, non-trivial grads
        o.zero_grad()
        ((ours_p @ Tensor(A)).tanh() ** 2).sum().backward()
        o.step()
        t.zero_grad()
        ((t_p @ torch.as_tensor(A)).tanh() ** 2).sum().backward()
        t.step()
    return np.max(np.abs(ours_p.data - t_p.detach().numpy()))


def test_sgd_momentum_nesterov_parity():
    err = _optimizer_parity(lambda p: optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=1e-3),
                            lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, weight_decay=1e-3))
    err2 = _optimizer_parity(lambda p: optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True),
                             lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True))
    assert err < 1e-12 and err2 < 1e-12, (err, err2)


def test_adam_parity():
    err = _optimizer_parity(lambda p: optim.Adam(p, lr=0.01, weight_decay=1e-2),
                            lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=1e-2))
    assert err < 1e-12, err


def test_adamw_parity():
    err = _optimizer_parity(lambda p: optim.AdamW(p, lr=0.01, weight_decay=0.1),
                            lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1))
    assert err < 1e-12, err


def test_init_scheme_statistics():
    rng = np.random.default_rng(0)
    w = nn.init.kaiming_normal((512, 512), rng)
    assert abs(w.std() - np.sqrt(2 / 512)) < 0.002
    w = nn.init.xavier_uniform((256, 512), rng)
    assert abs(w.std() - np.sqrt(2 / (256 + 512))) < 0.002


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    sys.exit(1 if run(TESTS) else 0)
