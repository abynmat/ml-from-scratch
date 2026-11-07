"""
Numerical-stability study (Author: Abyn Scaria).

  A. naive vs log-sum-exp softmax / cross-entropy as the logit scale grows   (float32)
  B. sigmoid saturation: exact-zero gradients in float32, and first-layer gradient vs init gain
  C. gradient norm per layer vs depth: tanh / sigmoid / relu / relu+LayerNorm
  D. finite-difference error vs step size h, float32 vs float64 (the V-shaped curve)

Writes results/stability_*.csv and four figures. Runs in well under a minute on CPU.
"""
import csv
import os
import sys

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import grad  # noqa: E402
import grad.functional as F  # noqa: E402
from grad import Tensor, nn  # noqa: E402

FIG, RES = os.path.join(ROOT, "figures"), os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def write_csv(name, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def nonfinite(a):
    return float(np.mean(~np.isfinite(a)))


# ------------------------------------------------------------------ A
def study_softmax():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((512, 10))
    y = rng.integers(0, 10, 512)
    rows = []
    for scale in np.logspace(0, 4, 33):
        ref = F.cross_entropy(Tensor(z * scale, dtype=np.float64), y).item()
        row = {"logit_scale": scale, "reference_loss_f64": ref}
        for tag, fn in (("naive", F.naive_cross_entropy), ("stable", F.cross_entropy)):
            x = Tensor((z * scale).astype(np.float32), requires_grad=True)
            with np.errstate(all="ignore"):
                loss = fn(x, y)
                loss.backward()
            L = loss.item()
            row[f"{tag}_loss_f32"] = L
            row[f"{tag}_rel_err"] = abs(L - ref) / abs(ref) if np.isfinite(L) else float("nan")
            row[f"{tag}_nonfinite_grad_frac"] = nonfinite(x.grad)
        rows.append(row)
    write_csv("stability_softmax.csv", rows)

    s = np.array([r["logit_scale"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.loglog(s, [r["reference_loss_f64"] for r in rows], "k-", lw=3, alpha=.3, label="float64 reference")
    for tag, c in (("naive", "tab:red"), ("stable", "tab:blue")):
        L = np.array([r[f"{tag}_loss_f32"] for r in rows])
        ok = np.isfinite(L)
        ax.loglog(s[ok], L[ok], "o-", c=c, ms=3, label=f"{tag} (float32)")
        if (~ok).any():
            ax.scatter(s[~ok], np.full((~ok).sum(), ax.get_ylim()[1] if ok.any() else 1),
                       marker="x", c=c, s=40, label=f"{tag}: loss is inf/NaN")
    ax.set_xlabel("logit scale  (logits = scale * N(0,1))")
    ax.set_ylabel("mean cross-entropy")
    ax.set_title("Cross-entropy value")
    ax.legend(fontsize=8)
    ax = axes[1]
    for tag, c in (("naive", "tab:red"), ("stable", "tab:blue")):
        ax.semilogx(s, [r[f"{tag}_nonfinite_grad_frac"] for r in rows], "o-", c=c, ms=3, label=tag)
    ax.set_xlabel("logit scale")
    ax.set_ylabel("fraction of gradient entries that are inf/NaN")
    ax.set_title("Gradient health")
    ax.legend()
    fig.suptitle("Naive softmax+log vs log-sum-exp (float32, 512x10 logits)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "stability_softmax_ce.png"), dpi=150)
    plt.close(fig)
    first_bad = next((r["logit_scale"] for r in rows if not np.isfinite(r["naive_loss_f32"])), None)
    first_bad_g = next((r["logit_scale"] for r in rows if r["naive_nonfinite_grad_frac"] > 0), None)
    worst_stable = max(r["stable_rel_err"] for r in rows)
    print(f"[A] naive CE first non-finite loss at scale {first_bad:.1f}; first non-finite grad at "
          f"{first_bad_g:.1f}; stable CE worst rel err {worst_stable:.1e}, non-finite grads "
          f"{max(r['stable_nonfinite_grad_frac'] for r in rows)}")
    return rows


# ------------------------------------------------------------------ B
def study_sigmoid():
    xs = np.linspace(0, 40, 161)
    exact = np.exp(-xs) / (1 + np.exp(-xs)) ** 2
    d = {}
    for dt in (np.float32, np.float64):
        x = Tensor(xs.astype(dt), requires_grad=True)
        x.sigmoid().sum().backward()
        d[dt] = x.grad.astype(np.float64)
    zero32 = xs[np.argmax(d[np.float32] == 0)]
    zero64 = xs[np.argmax(d[np.float64] == 0)] if (d[np.float64] == 0).any() else None

    # first-layer gradient norm of a 10-layer net as the init gain grows
    gains = np.logspace(-0.5, 1.0, 16)
    rows = []
    for act in ("sigmoid", "tanh"):
        for gain in gains:
            norms = []
            for seed in range(3):
                rng = np.random.default_rng(seed)
                layers = []
                for _ in range(10):
                    lin = nn.Linear(64, 64, init_scheme="lecun_normal", rng=rng)
                    lin.weight.data *= gain
                    layers += [lin, nn.Sigmoid() if act == "sigmoid" else nn.Tanh()]
                net = nn.Sequential(*layers, nn.Linear(64, 1, rng=rng))
                x = Tensor(rng.standard_normal((256, 64)).astype(np.float32))
                F.mse_loss(net(x), Tensor(rng.standard_normal((256, 1)).astype(np.float32))).backward()
                norms.append(np.linalg.norm(net[0].weight.grad))
            rows.append({"activation": act, "init_gain": gain, "first_layer_grad_norm": np.mean(norms)})
    write_csv("stability_sigmoid_gain.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.semilogy(xs, exact, "k-", lw=3, alpha=.3, label="exact  e^-x/(1+e^-x)^2")
    ax.semilogy(xs, np.where(d[np.float64] > 0, d[np.float64], np.nan), "--", c="tab:blue",
                label="autograd float64")
    ax.semilogy(xs, np.where(d[np.float32] > 0, d[np.float32], np.nan), "-", c="tab:red",
                label="autograd float32")
    ax.axvline(zero32, c="tab:red", ls=":", lw=1)
    ax.text(zero32 + 0.5, 1e-6, f"float32 grad is\nexactly 0 for x >= {zero32:.2f}", color="tab:red",
            fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("d sigmoid / dx")
    ax.set_title("Sigmoid saturation: 1 - sigmoid(x) rounds to 0")
    ax.legend(fontsize=8)
    ax = axes[1]
    for act, c in (("sigmoid", "tab:red"), ("tanh", "tab:green")):
        r = [q for q in rows if q["activation"] == act]
        ax.loglog([q["init_gain"] for q in r], [q["first_layer_grad_norm"] for q in r], "o-", c=c,
                  ms=3, label=act)
    ax.set_xlabel("init gain  (W = gain * N(0, 1/fan_in))")
    ax.set_ylabel("||dL/dW_1||  (layer 1 of 10)")
    ax.set_title("Layer-1 gradient of a 10-layer net vs init gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "stability_sigmoid_saturation.png"), dpi=150)
    plt.close(fig)
    print(f"[B] float32 sigmoid grad exactly zero from x={zero32:.2f}; float64 from x={zero64}")
    return zero32, zero64, rows


# ------------------------------------------------------------------ C
def study_depth(depth=30, width=128):
    configs = [("tanh", "uniform_fan_in", False), ("sigmoid", "uniform_fan_in", False),
               ("relu", "uniform_fan_in", False), ("relu", "kaiming_normal", False),
               ("relu", "uniform_fan_in", True)]
    rows = []
    curves = {}
    for act, scheme, use_ln in configs:
        label = f"{act}{' + LayerNorm' if use_ln else ''} ({scheme})"
        per_seed = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            layers, lins = [], []
            for _ in range(depth):
                lin = nn.Linear(width, width, init_scheme=scheme, rng=rng)
                lins.append(lin)
                layers.append(lin)
                if use_ln:
                    layers.append(nn.LayerNorm(width))
                layers.append({"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU}[act]())
            net = nn.Sequential(*layers, nn.Linear(width, 1, rng=rng))
            x = Tensor(rng.standard_normal((256, width)).astype(np.float32))
            t = Tensor(rng.standard_normal((256, 1)).astype(np.float32))
            F.mse_loss(net(x), t).backward()
            per_seed.append([np.linalg.norm(l.weight.grad) for l in lins])
        g = np.mean(per_seed, axis=0)
        curves[label] = g
        for i, v in enumerate(g):
            rows.append({"config": label, "layer": i + 1, "weight_grad_norm": v})
    write_csv("grad_norm_per_layer.csv", rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, g in curves.items():
        ax.semilogy(np.arange(1, depth + 1), g, "o-", ms=3, label=label)
    ax.set_xlabel("layer (1 = closest to input)")
    ax.set_ylabel("||dL/dW_l||  (mean of 3 seeds)")
    ax.set_title(f"Gradient norm per layer, {depth}-layer MLP, width {width}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "grad_norm_per_layer.png"), dpi=150)
    plt.close(fig)
    summary = {k: (v[0], v[-1], v[-1] / v[0] if v[0] > 0 else float("inf")) for k, v in curves.items()}
    for k, (first, last, ratio) in summary.items():
        print(f"[C] {k:<40} layer1 {first:.2e}  layer{depth} {last:.2e}  last/first {ratio:.1e}")
    return summary


# ------------------------------------------------------------------ D
def study_fd_step():
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((6, 16)) / np.sqrt(6)
    W2 = rng.standard_normal((16, 1)) / 4
    x0 = rng.standard_normal((4, 6))

    def f(x, dt):
        h1 = (Tensor(x, dtype=dt) @ Tensor(W1, dtype=dt)).tanh()
        return ((h1 @ Tensor(W2, dtype=dt)).sigmoid()).sum()

    xt = Tensor(x0, requires_grad=True)
    f(xt.data, np.float64)  # warm
    xt = Tensor(x0.copy(), requires_grad=True)
    (((xt @ Tensor(W1)).tanh() @ Tensor(W2)).sigmoid()).sum().backward()
    g_true = xt.grad                                   # verified analytic gradient (float64)
    coords = list(np.ndindex(x0.shape))
    hs = np.logspace(-14, 0, 57)
    rows = []
    for dt in (np.float32, np.float64):
        for h in hs:
            errs_c, errs_f = [], []
            for c in coords:
                xp, xm = x0.astype(dt), x0.astype(dt)
                xp = xp.copy(); xm = xm.copy()
                xp[c] += dt(h)
                xm[c] -= dt(h)
                fp, fm = float(f(xp, dt).data), float(f(xm, dt).data)
                f0 = float(f(x0.astype(dt), dt).data)
                errs_c.append(abs((fp - fm) / (2 * h) - g_true[c]))
                errs_f.append(abs((fp - f0) / h - g_true[c]))
            rows.append({"dtype": np.dtype(dt).name, "h": h, "central_median_abs_err": np.median(errs_c),
                         "forward_median_abs_err": np.median(errs_f)})
    write_csv("stability_fd_step.csv", rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for dtn, c in (("float32", "tab:red"), ("float64", "tab:blue")):
        r = [q for q in rows if q["dtype"] == dtn]
        h = np.array([q["h"] for q in r])
        ec = np.array([q["central_median_abs_err"] for q in r])
        ef = np.array([q["forward_median_abs_err"] for q in r])
        ax.loglog(h, ec, "o-", c=c, ms=3, label=f"{dtn} central")
        ax.loglog(h, ef, "--", c=c, alpha=.6, label=f"{dtn} forward")
        eps = np.finfo(dtn).eps
        ax.axvline(eps ** (1 / 3), c=c, ls=":", lw=1)
        best = h[np.argmin(ec)]
        print(f"[D] {dtn}: best central h={best:.1e} (theory eps^(1/3)={eps ** (1/3):.1e}), "
              f"min median err {ec.min():.1e}; forward best h={h[np.argmin(ef)]:.1e} err {ef.min():.1e}")
    ax.set_xlabel("step size h")
    ax.set_ylabel("median |FD - autograd|  over 24 inputs")
    ax.set_title("Finite-difference error vs step size\n(dotted lines: eps^(1/3), the central-difference optimum)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "stability_fd_step_size.png"), dpi=150)
    plt.close(fig)
    return rows


if __name__ == "__main__":
    grad.set_default_dtype(np.float32)
    study_softmax()
    study_sigmoid()
    study_depth()
    study_fd_step()
