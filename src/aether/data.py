"""Load Fashion-MNIST images AND labels for the referential game."""

import gzip
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def _read_images(fn):
    with gzip.open(os.path.join(DATA, fn), "rb") as f:
        buf = f.read()
    n = int.from_bytes(buf[4:8], "big"); r = int.from_bytes(buf[8:12], "big")
    c = int.from_bytes(buf[12:16], "big")
    return np.frombuffer(buf[16:], np.uint8).reshape(n, r, c)


def _read_labels(fn):
    with gzip.open(os.path.join(DATA, fn), "rb") as f:
        buf = f.read()
    return np.frombuffer(buf[8:], np.uint8).copy()


def load(split="train", device="cpu"):
    if split == "train":
        imgs, labs = _read_images("train-images-idx3-ubyte.gz"), _read_labels("train-labels-idx1-ubyte.gz")
    else:
        imgs, labs = _read_images("t10k-images-idx3-ubyte.gz"), _read_labels("t10k-labels-idx1-ubyte.gz")
    x = torch.from_numpy(imgs.copy()).float().div_(255.0).unsqueeze(1).to(device)  # (N,1,28,28)
    y = torch.from_numpy(labs).long().to(device)
    return x, y


class Referential:
    """Builds referential-game rounds from a labelled image pool.

    A round: the sender sees a target image of class c. The receiver sees K
    candidate images — one is a DIFFERENT image of class c (the correct answer),
    the other K-1 are images of OTHER classes. To win, the sender's message must
    convey enough about class c for the receiver to pick it out. Because the
    correct candidate is a *different* instance, the agents can't cheat on pixels;
    they must communicate something class-level and abstract.
    """

    def __init__(self, x, y, K=8, n_classes=10, device="cpu"):
        self.x, self.y, self.K, self.C, self.device = x, y, K, n_classes, device
        # Build a (C, M) table of image indices per class (M = min class count) so
        # sampling is a pure gather — no Python loops in the hot path.
        per = [torch.nonzero(y == c, as_tuple=True)[0] for c in range(n_classes)]
        M = min(len(p) for p in per)
        self.M = M
        self.class_idx = torch.stack([p[torch.randperm(len(p))[:M]] for p in per]).to(device)  # (C,M)

    def batch(self, B, gen=None):
        C, K, M, dev = self.C, self.K, self.M, self.device
        target = torch.randint(C, (B,), device=dev)
        answer = torch.randint(K, (B,), device=dev)
        # distractor classes: offset trick guarantees every slot != target class...
        cand_cls = (target[:, None] + 1 + torch.randint(C - 1, (B, K), device=dev)) % C
        # ...then the answer slot is set to the target class.
        cand_cls[torch.arange(B, device=dev), answer] = target
        cand_img_idx = self.class_idx[cand_cls, torch.randint(M, (B, K), device=dev)]  # (B,K)
        cands = self.x[cand_img_idx.reshape(-1)].reshape(B, K, 1, 28, 28)
        sender_idx = self.class_idx[target, torch.randint(M, (B,), device=dev)]
        sender_img = self.x[sender_idx]                                                # (B,1,28,28)
        return sender_img, cands, answer, target
