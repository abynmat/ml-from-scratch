"""
Performance study (Author: Abyn Scaria): forward+backward wall time of the same
MLP (8 -> 32 -> 32 -> 1, tanh, MSE loss) in three engines as batch size grows:

  * grad        -- this repo's tensor engine (NumPy, float32)
  * PyTorch     -- torch 2.x on CPU, float32, 2 threads
  * scalar      -- grad/scalar.py, one Python object per number (micrograd-style)

All three run on CPU with BLAS limited to 2 threads. Each point is the median of
repeated runs (at least 5, or ~0.3 s of samples). Writes results/timing.csv and
figures/timing_loglog.png.
"""
import csv
import os
import sys
import time

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

import grad.functional as F  # noqa: E402
from grad import Tensor, nn  # noqa: E402
from grad.scalar import Scalar, scalar_forward, scalar_mlp  # noqa: E402

SIZES = [8, 32, 32, 1]
BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]
SCALAR_MAX = 256


def bench(fn, min_reps=5, budget=0.3):
    fn()  # warm-up
    times, t_start = [], time.perf_counter()
    while len(times) < min_reps or time.perf_counter() - t_start < budget:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        if len(times) >= 200:
            break
    return float(np.median(times))


def main():
    rng = np.random.default_rng(0)
    ours = nn.Sequential(nn.Linear(8, 32, rng=rng), nn.Tanh(), nn.Linear(32, 32, rng=rng),
                         nn.Tanh(), nn.Linear(32, 1, rng=rng))
    theirs = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32),
                                 torch.nn.Tanh(), torch.nn.Linear(32, 1))
    scal = scalar_mlp(SIZES)
    rows = []
    for B in BATCHES:
        X = rng.standard_normal((B, 8)).astype(np.float32)
        Y = rng.standard_normal((B, 1)).astype(np.float32)
        tX, tY = torch.as_tensor(X), torch.as_tensor(Y)

        def run_ours():
            ours.zero_grad()
            F.mse_loss(ours(Tensor(X)), Tensor(Y)).backward()

        def run_torch():
            theirs.zero_grad()
            torch.nn.functional.mse_loss(theirs(tX), tY).backward()

        def run_scalar():
            loss = Scalar(0.0)
            for x, y in zip(X.tolist(), Y[:, 0].tolist()):
                d = scalar_forward(scal, x)[0] + (-y)
                loss = loss + d * d
            (loss * (1.0 / B)).backward()

        row = {"batch": B, "grad_s": bench(run_ours), "torch_s": bench(run_torch),
               "scalar_s": bench(run_scalar, min_reps=3, budget=0.5) if B <= SCALAR_MAX else ""}
        rows.append(row)
        print(f"B={B:>6}  grad {row['grad_s']*1e3:9.3f} ms  torch {row['torch_s']*1e3:9.3f} ms  "
              f"scalar {row['scalar_s']*1e3 if row['scalar_s'] != '' else float('nan'):10.1f} ms",
              flush=True)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "timing.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b = np.array([r["batch"] for r in rows])
    ax.loglog(b, [r["grad_s"] * 1e3 for r in rows], "o-", label="grad (this repo, NumPy tensors)")
    ax.loglog(b, [r["torch_s"] * 1e3 for r in rows], "s-", label="PyTorch CPU")
    sb = [r for r in rows if r["scalar_s"] != ""]
    ax.loglog([r["batch"] for r in sb], [r["scalar_s"] * 1e3 for r in sb], "^-",
              label="scalar graph (micrograd-style)")
    ax.set_xlabel("batch size")
    ax.set_ylabel("forward + backward time (ms, median)")
    ax.set_title("MLP 8-32-32-1, tanh, MSE  |  CPU, 2 threads, float32")
    ax.grid(True, which="both", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "timing_loglog.png"), dpi=150)
    return rows


if __name__ == "__main__":
    main()
