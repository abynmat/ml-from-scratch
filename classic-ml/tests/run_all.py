"""Run every tests/test_*.py module with plain python:  python3 tests/run_all.py

Author: Abyn Scaria
"""
import glob
import importlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _harness  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    total_failed = 0
    for path in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        print(name)
        total_failed += _harness.run(vars(importlib.import_module(name)))
    print(f"\n{'ALL PASSED' if total_failed == 0 else f'{total_failed} FAILED'} in {time.time() - t0:.1f}s")
    raise SystemExit(1 if total_failed else 0)
