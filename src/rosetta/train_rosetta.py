"""Train Rosetta adapters so every (sender i -> receiver j) pair interoperates.

Base models are frozen; only the 2N small adapters learn. Objective: the mean
reconstruction error across ALL i,j pairs, over a range of channel SNRs.
"""

import os
import random
import time
import torch

from model import LatentRadio, psnr
from data import load_fashion
from rosetta import Rosetta

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/Users/priyankarchakraborty/Trading/latent-radio/models"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEEDS = [0, 1, 2]           # three independently-trained "vendors"
STEPS, BATCH, LR = 2500, 128, 8e-4


def load_bases():
    ms = []
    for s in SEEDS:
        m = LatentRadio(k=128)
        m.load_state_dict(torch.load(f"{BASE}/model_{s}.pt", map_location=DEVICE)["state_dict"])
        ms.append(m.to(DEVICE).eval())
    return ms


def main():
    torch.manual_seed(0); random.seed(0)
    bases = load_bases()
    ros = Rosetta(bases).to(DEVICE)
    xtr = load_fashion("train", device=DEVICE)
    n = xtr.shape[0]
    opt = torch.optim.Adam([p for p in ros.parameters() if p.requires_grad], lr=LR)
    print(f"training {2*len(SEEDS)} adapters over {len(SEEDS)}x{len(SEEDS)} pairs "
          f"({sum(p.numel() for p in ros.parameters() if p.requires_grad):,} params)")

    t0 = time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(n, (BATCH,), device=DEVICE)
        x = xtr[idx]
        snr = random.uniform(-2.0, 14.0)
        loss = 0.0
        for i in range(ros.n):
            r_rx = ros.transmit(x, i, snr)
            for j in range(ros.n):
                out = ros.receive(r_rx, j)
                loss = loss + torch.mean((out - x) ** 2)
        loss = loss / (ros.n * ros.n)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")

    torch.save({"to_ref": ros.to_ref.state_dict(), "from_ref": ros.from_ref.state_dict(),
                "seeds": SEEDS}, os.path.join(HERE, "rosetta_adapters.pt"))
    print("saved rosetta_adapters.pt")


if __name__ == "__main__":
    main()
