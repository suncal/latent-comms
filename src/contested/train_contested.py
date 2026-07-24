"""
Train two shared-model transceivers on Fashion-MNIST:
  * contested  — trained through the full contested channel (fade + jam + AWGN).
  * awgn_only  — trained through AWGN alone (the honest ablation: this is what you
                 get if you design for clean noise and ignore the threat).

Both reuse the Latent Radio encoder/decoder; only the training channel differs.
"""

import os
import random
import time
import torch

from data import load_fashion
from model import LatentRadio, psnr, count_params
from contested import contested, sample_train_channel

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
K, EPOCHS, BATCH, LR = 128, 14, 128, 8e-4


def train(mode, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    xtr = load_fashion("train", device=DEVICE)
    xva = load_fashion("test", limit=2000, device=DEVICE)
    n = xtr.shape[0]
    m = LatentRadio(k=K).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    print(f"[{mode}] device={DEVICE} params={count_params(m):,}")
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(n, device=DEVICE); t0 = time.time()
        for i in range(0, n, BATCH):
            xb = xtr[perm[i:i + BATCH]]
            if mode == "contested":
                snr, erase, fade = sample_train_channel()
            else:  # awgn_only
                snr, erase, fade = random.uniform(-4, 14), 0.0, 0.0
            out = m.dec(contested(m.enc(xb), snr, erase, fade))
            loss = torch.mean((out - xb) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 4 == 0 or ep == EPOCHS:
            m.eval()
            with torch.no_grad():
                clean = psnr(xva, m.dec(contested(m.enc(xva), 10, 0, 0))).mean().item()
                jam = psnr(xva, m.dec(contested(m.enc(xva), 4, 0.35, 0.3))).mean().item()
            m.train()
            print(f"  ep{ep:2d} {time.time()-t0:4.1f}s  clean {clean:5.1f}dB  jammed {jam:5.1f}dB")
    path = os.path.join(HERE, f"model_{mode}.pt")
    torch.save({"state_dict": m.state_dict(), "k": K}, path)
    print("  saved", path)


if __name__ == "__main__":
    train("contested", seed=0)
    train("awgn_only", seed=0)
