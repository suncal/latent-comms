"""Train one JSCC transceiver over the full standard channel mix (fading + jamming)."""

import os
import random
import time
import torch

from data import load
from model import ContestedJSCC, psnr, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS, BATCH, LR = 18, 128, 6e-4


def sample_channel():
    kind = "rayleigh" if random.random() < 0.5 else "rician"
    K = random.uniform(2.0, 8.0)
    snr = random.uniform(-2.0, 18.0)
    if random.random() < 0.7:
        rho, jsr = random.uniform(0.05, 0.5), random.uniform(0.0, 12.0)
    else:
        rho, jsr = 0.0, None
    return dict(snr_db=snr, kind=kind, K=K, jam_rho=rho, jsr_db=jsr, block_len=8)


def main():
    torch.manual_seed(0); random.seed(0)
    xtr, _ = load("train", device=DEVICE)
    xva, _ = load("test", device=DEVICE)
    n = xtr.shape[0]
    m = ContestedJSCC().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    print(f"device={DEVICE} params={count_params(m):,} train={n}")
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(n, device=DEVICE); t0 = time.time()
        for i in range(0, n, BATCH):
            xb = xtr[perm[i:i + BATCH]]
            out, _ = m(xb, **sample_channel())
            loss = torch.mean((out - xb) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 0 or ep == EPOCHS:
            m.eval()
            with torch.no_grad():
                ray = psnr(xva, m(xva, snr_db=10, kind="rayleigh")[0]).mean().item()
                jam = psnr(xva, m(xva, snr_db=6, kind="rayleigh", jam_rho=0.3, jsr_db=8)[0]).mean().item()
            m.train()
            print(f"  ep{ep:2d} {time.time()-t0:4.1f}s  rayleigh@10dB {ray:5.1f}dB  +jam {jam:5.1f}dB")
    torch.save({"state_dict": m.state_dict(), "m": m.m}, os.path.join(HERE, "jscc.pt"))
    print("saved jscc.pt")


if __name__ == "__main__":
    main()
