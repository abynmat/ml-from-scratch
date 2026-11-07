"""Tiny test harness so the suite runs with plain `python3` (no pytest needed).

Author: Abyn Scaria
"""
import os
import sys
import time
import traceback
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# NumPy 2.x on macOS Accelerate raises spurious "encountered in matmul" warnings on valid inputs
warnings.filterwarnings("ignore", message=".*encountered in matmul")


def close(a, b, tol, what):
    """Assert |a - b| <= tol elementwise, with a readable message."""
    import numpy as np
    diff = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    assert diff <= tol, f"{what}: max |diff| = {diff:.3g} > tol {tol:.3g}"


def run(module_globals):
    """Run every top-level function named test_* in the calling module. Returns #failures."""
    tests = [(n, f) for n, f in module_globals.items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"  PASS  {name}  ({time.time() - t0:.2f}s)")
        except Exception:  # noqa: BLE001 - report every failure, keep going
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    return failed
