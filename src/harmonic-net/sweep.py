"""
Noise sweep: HRM-full vs GRU on frequency-ID across rising noise levels.

This is the headline capability plot. Both models are trained identically at each
noise level; we report final test accuracy on a fixed 2000-example test set. The
gap between them widens as noise rises, because HRM's tuned-resonator prior
integrates energy at the target frequency and rejects broadband noise, while the
GRU has no such built-in prior.

Writes results_sweep.json.
"""

import json
import torch
import torch.nn.functional as F

import data_freq as D
from model import HRM, GRUBaseline

NOISES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SEQ_LEN = 100
STEPS = 900
BATCH = 128
LR = 3e-3
K = D.N_CLASSES


def train_eval(make_model, noise):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    eg = torch.Generator().manual_seed(99999)
    m = make_model()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for _ in range(STEPS):
        x, y = D.make_batch(BATCH, SEQ_LEN, noise=noise, generator=g)
        F.cross_entropy(m(x), y).backward()
        opt.step()
        opt.zero_grad()
    m.eval()
    with torch.no_grad():
        x, y = D.make_batch(2000, SEQ_LEN, noise=noise, generator=eg)
        return (m(x).argmax(-1) == y).float().mean().item()


def main():
    makers = {
        "HRM-full": lambda: HRM(d_in=1, n_channels=48, out_dim=K),
        "GRU": lambda: GRUBaseline(d_in=1, hidden=48, out_dim=K),
    }
    out = {"noises": NOISES, "chance": D.chance_accuracy(), "series": {}}
    for name, mk in makers.items():
        accs = []
        for nz in NOISES:
            a = train_eval(mk, nz)
            accs.append(round(a, 4))
            print(f"{name:9s} noise={nz:.1f}  acc={a:.3f}")
        out["series"][name] = accs
        print()
    with open("results_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote results_sweep.json")


if __name__ == "__main__":
    main()
