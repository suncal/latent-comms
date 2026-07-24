"""Load Fashion-MNIST (and optionally MNIST digits) from raw IDX gz files."""

import gzip
import os
import urllib.request
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic = int.from_bytes(buf[0:4], "big")
    assert magic == 2051, f"bad image magic {magic}"
    n = int.from_bytes(buf[4:8], "big")
    r = int.from_bytes(buf[8:12], "big")
    c = int.from_bytes(buf[12:16], "big")
    arr = np.frombuffer(buf[16:], dtype=np.uint8).reshape(n, r, c)
    return arr


def load_fashion(split="train", limit=None, device="cpu"):
    fn = "train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz"
    arr = _read_idx_images(os.path.join(DATA, fn))
    if limit:
        arr = arr[:limit]
    x = torch.from_numpy(arr.copy()).float().div_(255.0).unsqueeze(1)  # (N,1,28,28)
    return x.to(device)


_MNIST_URLS = {
    "t10k-images-idx3-ubyte.gz":
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
}


def load_mnist_test(limit=2000, device="cpu"):
    """Digits — used only as an out-of-distribution test set."""
    path = os.path.join(DATA, "mnist-t10k-images-idx3-ubyte.gz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(_MNIST_URLS["t10k-images-idx3-ubyte.gz"], path)
    arr = _read_idx_images(path)[:limit]
    x = torch.from_numpy(arr.copy()).float().div_(255.0).unsqueeze(1)
    return x.to(device)


if __name__ == "__main__":
    tr = load_fashion("train")
    te = load_fashion("test")
    print("train", tuple(tr.shape), "test", tuple(te.shape),
          "range", float(tr.min()), float(tr.max()))
