"""Run every test file as a subprocess (they set different default dtypes). Author: Abyn Scaria."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
failed = 0
for f in ("test_ops.py", "test_nn.py"):
    print(f"=== {f}")
    r = subprocess.run([sys.executable, os.path.join(HERE, f)], capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    print("\n".join(l for l in lines if l.startswith("FAIL")) or "(no failures)")
    print(lines[-1] if lines else r.stderr[-2000:])
    failed += r.returncode != 0
sys.exit(1 if failed else 0)
