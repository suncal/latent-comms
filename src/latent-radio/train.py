"""
Train a Latent Radio transceiver end-to-end, SNR-adaptive.

Each batch is transmitted at a random SNR drawn from a range, so a SINGLE trained
model is robust across channel conditions — which is what produces graceful
degradation at test time (rather than a model tuned to one exact SNR).

Usage: python3 train.py [seed]     # writes models/model_<seed>.pt
Train two seeds (0 and 1) to enable the "independently trained models don't
interoperate" failure test in evaluate.py.
"""

import os
import sys
import time
import torch

from data import load_fashion
from model import LatentRadio, psnr, count_params

K = 128
EPOCHS = 12
BATCH = 128
LR = 8e-4
SNR_TRAIN_RANGE = (-2.0, 12.0)  # dB, sampled uniformly per batch
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))


def sample_snr():
    lo, hi = SNR_TRAIN_RANGE
    return float(torch.empty(1).uniform_(lo, hi).item())


def main(seed=0):
    torch.manual_seed(seed)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)

    xtr = load_fashion("train", device=DEVICE)
    xva = load_fashion("test", limit=2000, device=DEVICE)
    n = xtr.shape[0]

    model = LatentRadio(k=K).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"seed={seed}  device={DEVICE}  params={count_params(model):,}  "
          f"train={n}  k={K}")

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        t0 = time.time()
        for i in range(0, n, BATCH):
            xb = xtr[perm[i:i + BATCH]]
            out = model(xb, sample_snr())
            loss = torch.mean((out - xb) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # validate across a few SNRs
        model.eval()
        with torch.no_grad():
            msg = []
            for s in (0.0, 6.0, 12.0):
                p = psnr(xva, model(xva, s)).mean().item()
                msg.append(f"{s:.0f}dB:{p:.1f}")
        print(f"  ep {ep:2d}  {time.time()-t0:5.1f}s  valPSNR " + "  ".join(msg))

    path = os.path.join(HERE, "models", f"model_{seed}.pt")
    torch.save({"state_dict": model.state_dict(), "k": K}, path)
    print("saved", path)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
