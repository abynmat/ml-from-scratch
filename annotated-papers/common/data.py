"""IDX-format loaders for MNIST / Fashion-MNIST (no torchvision needed).

Author: Abyn Scaria. Whole datasets are small (60k x 784 bytes), so they are
loaded straight onto the training device as float tensors; batching is then a
cheap index into a resident tensor instead of a DataLoader.
"""
import gzip
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def _read_idx(path):
    """Parse one IDX file: magic (2 zero bytes, dtype code, ndim), dims, raw bytes."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(name="mnist", device="cpu", n_train=None):
    """Return (x_train, y_train, x_test, y_test); x in [0, 1], shape (N, 1, 28, 28)."""
    d = os.path.join(DATA, name)
    if not os.path.exists(os.path.join(d, "t10k-labels-idx1-ubyte.gz")):
        raise FileNotFoundError(f"{d} missing; run: python3 data/download.py {name}")
    xs = []
    for split in ("train", "t10k"):
        x = _read_idx(os.path.join(d, f"{split}-images-idx3-ubyte.gz"))
        y = _read_idx(os.path.join(d, f"{split}-labels-idx1-ubyte.gz"))
        xs.append(torch.tensor(x, dtype=torch.float32).div_(255.0).unsqueeze(1))
        xs.append(torch.tensor(y, dtype=torch.long))
    xtr, ytr, xte, yte = xs
    if n_train is not None:
        xtr, ytr = xtr[:n_train], ytr[:n_train]
    return xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)
