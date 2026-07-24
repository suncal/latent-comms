"""Train the analog JSCC layer of the hybrid at k_a=64 complex uses (SNR-adaptive)."""

import os
import random
import time
import torch

from data import load
from model import AdaptiveJSCC, psnr, count_params
from channel import pack, power_normalize, cmul, block_fading, awgn

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
KA = int(sys.argv[1]) if len(sys.argv) > 1 else 64   # analog complex uses
EPOCHS, BATCH, LR, BL = 20, 128, 6e-4, 8


def sample_ch():
    kind = "rayleigh" if random.random() < 0.5 else "rician"
    return kind, random.uniform(2.0, 8.0), random.uniform(-2.0, 18.0)


def main():
    torch.manual_seed(0); random.seed(0)
    xtr, _ = load("train", device=DEVICE); xva, _ = load("test", device=DEVICE)
    n = xtr.shape[0]
    m = AdaptiveJSCC(m=KA).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    print(f"analog-{KA}  params={count_params(m):,}")
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(n, device=DEVICE); t0 = time.time()
        for i in range(0, n, BATCH):
            xb = xtr[perm[i:i + BATCH]]
            kind, K, snr = sample_ch()
            z, cond = m.encode(xb, snr)
            s = power_normalize(pack(z)); B, mm, _ = s.shape
            h = block_fading(B, mm, BL, xb.device, kind, K)
            y = cmul(h, s); y, _ = awgn(y, snr)
            out = m.dec(torch.cat([y, h], -1), cond)
            loss = torch.mean((out - xb) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0 or ep == EPOCHS:
            m.eval()
            with torch.no_grad():
                hi = psnr(xva, m(xva, snr_db=18, kind="rayleigh")[0]).mean().item()
                lo = psnr(xva, m(xva, snr_db=0, kind="rayleigh")[0]).mean().item()
            m.train()
            print(f"  ep{ep:2d} {time.time()-t0:4.1f}s  R@18 {hi:5.1f}  R@0 {lo:5.1f}")
    out = os.path.join(HERE, f"model_analog{KA}.pt")
    torch.save({"state_dict": m.state_dict(), "m": KA}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
