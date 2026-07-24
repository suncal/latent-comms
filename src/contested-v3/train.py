"""Train the SNR-adaptive JSCC, and a reactive-jam-hardened version."""

import os
import sys
import random
import time
import torch

from data import load
from model import AdaptiveJSCC, psnr, count_params
from channel import pack, power_normalize, cmul, block_fading, awgn, partial_band_jam
from jammers import follower_jam

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS, BATCH, LR, BL = 18, 128, 6e-4, 8


def sample_ch():
    kind = "rayleigh" if random.random() < 0.5 else "rician"
    K = random.uniform(2.0, 8.0)
    snr = random.uniform(-2.0, 18.0)
    if random.random() < 0.7:
        rho, jsr = random.uniform(0.05, 0.5), random.uniform(0.0, 12.0)
    else:
        rho, jsr = 0.0, None
    return kind, K, snr, rho, jsr


def channel_feats(z, snr, kind, K, rho, jsr, follower):
    s = power_normalize(pack(z)); B, m, _ = s.shape
    h = block_fading(B, m, BL, z.device, kind, K)
    y = cmul(h, s); y, _ = awgn(y, snr)
    if follower and jsr is not None:
        y = follower_jam(y, s, jsr, rho=random.uniform(0.2, 0.4))
    elif rho > 0 and jsr is not None:
        y, _ = partial_band_jam(y, rho, jsr)
    return torch.cat([y, h], dim=-1)


def train(robust, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    xtr, _ = load("train", device=DEVICE); xva, _ = load("test", device=DEVICE)
    n = xtr.shape[0]
    m = AdaptiveJSCC().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    tag = "robust" if robust else "adaptive"
    print(f"[{tag}] params={count_params(m):,}")
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(n, device=DEVICE); t0 = time.time()
        for i in range(0, n, BATCH):
            xb = xtr[perm[i:i + BATCH]]
            kind, K, snr, rho, jsr = sample_ch()
            use_follower = robust and random.random() < 0.5
            z, cond = m.encode(xb, snr)
            feats = channel_feats(z, snr, kind, K, rho, jsr, use_follower)
            out = m.dec(feats, cond)
            loss = torch.mean((out - xb) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 6 == 0 or ep == EPOCHS:
            m.eval()
            with torch.no_grad():
                hi = psnr(xva, m(xva, snr_db=18, kind="rayleigh")[0]).mean().item()
                lo = psnr(xva, m(xva, snr_db=0, kind="rayleigh")[0]).mean().item()
            m.train()
            print(f"  ep{ep:2d} {time.time()-t0:4.1f}s  R@18dB {hi:5.1f}  R@0dB {lo:5.1f}")
    path = os.path.join(HERE, f"model_{tag}.pt")
    torch.save({"state_dict": m.state_dict(), "m": m.m}, path)
    print("saved", path)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("adaptive", "both"):
        train(robust=False)
    if which in ("robust", "both"):
        train(robust=True)
