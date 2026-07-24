"""Load EuroSAT (Sentinel-2 RGB satellite imagery), cached as a tensor."""

import os
import glob
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IMG = 32
CACHE = os.path.join(DATA, f"eurosat_{IMG}.npz")
CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


def build_cache():
    files = sorted(glob.glob(os.path.join(DATA, "2750", "*", "*.jpg")))
    assert files, "EuroSAT images not found"
    xs, ys = [], []
    cls_index = {c: i for i, c in enumerate(CLASSES)}
    for f in files:
        cls = os.path.basename(os.path.dirname(f))
        im = Image.open(f).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
        xs.append(np.asarray(im, np.uint8))
        ys.append(cls_index[cls])
    x = np.stack(xs)                    # (N,32,32,3) uint8
    y = np.array(ys, np.int64)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    np.savez_compressed(CACHE, x=x, y=y)
    print(f"cached {len(x)} images -> {CACHE}")


def load(split="train", device="cpu", n_test=3000):
    if not os.path.exists(CACHE):
        build_cache()
    d = np.load(CACHE)
    x, y = d["x"], d["y"]
    if split == "train":
        x, y = x[n_test:], y[n_test:]
    else:
        x, y = x[:n_test], y[:n_test]
    xt = torch.from_numpy(x).float().div_(255.0).permute(0, 3, 1, 2).contiguous().to(device)  # (N,3,32,32)
    yt = torch.from_numpy(y).to(device)
    return xt, yt


if __name__ == "__main__":
    build_cache()
    x, y = load("test")
    print("test", tuple(x.shape), "range", float(x.min()), float(x.max()))
