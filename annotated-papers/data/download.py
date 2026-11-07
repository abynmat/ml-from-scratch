"""Download MNIST and Fashion-MNIST in their original IDX format.

Author: Abyn Scaria. torchvision is deliberately not required: the four IDX
files per dataset are fetched from public mirrors and parsed by
common/data.py with numpy.

    python3 data/download.py            # both datasets, ~60 MB total
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

MIRRORS = {
    # Public mirror of Yann LeCun's MNIST files (same bytes, same names).
    "mnist": "https://storage.googleapis.com/cvdf-datasets/mnist/",
    # Fashion-MNIST is distributed from its own GitHub repository.
    "fashion": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
}
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download(name):
    out_dir = os.path.join(HERE, name)
    os.makedirs(out_dir, exist_ok=True)
    for f in FILES:
        dst = os.path.join(out_dir, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        url = MIRRORS[name] + f
        print(f"  fetching {url}")
        urllib.request.urlretrieve(url, dst)
    print(f"{name}: ok -> {out_dir}")


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(MIRRORS)):
        download(n)
