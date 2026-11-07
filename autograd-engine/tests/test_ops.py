"""
Per-op gradient tests vs PyTorch and finite differences (Author: Abyn Scaria).
Run as plain Python:  python3 tests/test_ops.py   (exit code 1 on any failure)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

from gradcheck import run_case  # noqa: E402
from grad import Tensor, no_grad, unbroadcast  # noqa: E402
from tests.op_specs import CASES  # noqa: E402

TOL_TORCH = 1e-10     # float64: both engines should agree to rounding
TOL_FD = 1e-6         # central differences with h=1e-6 are good to ~1e-9..1e-7


def make_case_test(case):
    def test():
        for seed in (0, 1):
            r = run_case(case, seed)
            assert r["fwd_abs_vs_torch"] < TOL_TORCH, f"forward mismatch {r}"
            assert r["grad_rel_vs_torch"] < TOL_TORCH, f"grad vs torch {r}"
            assert r["grad_rel_vs_fd"] < TOL_FD, f"grad vs finite differences {r}"
    test.__name__ = f"test_op_{case[0]}"
    return test


def test_unbroadcast_shapes():
    g = np.ones((5, 3, 4))
    assert unbroadcast(g, (3, 4)).shape == (3, 4) and unbroadcast(g, (3, 4))[0, 0] == 5
    assert unbroadcast(g, (1, 4)).shape == (1, 4) and unbroadcast(g, (1, 4))[0, 0] == 15
    assert unbroadcast(g, (5, 1, 1))[0, 0, 0] == 12


def test_grad_accumulates_over_reuse():
    # y = x*x + x  uses x three times -> dy/dx = 2x + 1
    x = Tensor(np.array([3.0]), requires_grad=True)
    (x * x + x).sum().backward()
    assert np.allclose(x.grad, [7.0])


def test_grad_accumulates_across_backward_calls():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    (x * 2).sum().backward()
    (x * 3).sum().backward()
    assert np.allclose(x.grad, [5.0, 5.0])


def test_no_grad_records_nothing():
    x = Tensor(np.ones(3), requires_grad=True)
    with no_grad():
        y = (x * 2).exp()
    assert not y.requires_grad and y.is_leaf


def test_deep_graph_no_recursion_error():
    x = Tensor(np.array([0.5]), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.sum().backward()
    assert np.allclose(x.grad, 1.0001 ** 5000)


def test_softmax_stable_at_large_logits():
    x = Tensor(np.array([[1000.0, 0.0, -1000.0]], dtype=np.float32), requires_grad=True)
    s = x.softmax(-1)
    ls = x.log_softmax(-1)
    (ls[0, 1]).backward()
    assert np.all(np.isfinite(s.data)) and np.all(np.isfinite(ls.data))
    assert np.all(np.isfinite(x.grad))


TESTS = [make_case_test(c) for c in CASES] + [
    test_unbroadcast_shapes, test_grad_accumulates_over_reuse,
    test_grad_accumulates_across_backward_calls, test_no_grad_records_nothing,
    test_deep_graph_no_recursion_error, test_softmax_stable_at_large_logits,
]


def run(tests):
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run(TESTS) else 0)
