"""
Train the Aether agents and run the full experiment suite.

E1  Bandwidth sweep — accuracy vs message size d (how few numbers they need).
E2  Noise curve — the main model's accuracy vs channel SNR (graceful), against a
    classical digital baseline that transmits the target class over the same
    channel at Shannon capacity (a cliff).
E3  Desync failure — sender from run A + receiver from run B (a private language).

Writes results.json.
"""

import os
import json
import math
import random
import time
import torch
import torch.nn.functional as F

from data import load, Referential
from model import Aether, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
K = 8
NOISE_RANGE = (-6.0, 18.0)
SNR_GRID = [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 16, 20]
BATCH = 128

_x, _y = load("train", device=DEVICE)
_xt, _yt = load("test", device=DEVICE)
_train = Referential(_x, _y, K=K, device=DEVICE)
_test = Referential(_xt, _yt, K=K, device=DEVICE)


def _snr():
    return random.uniform(*NOISE_RANGE)


def evaluate(model, snr_db, n=3000, draws=3):
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(draws):
            si, cn, ans, _ = _test.batch(n)
            accs.append((model(si, cn, snr_db).argmax(1) == ans).float().mean().item())
    model.train()
    return sum(accs) / len(accs)


def train_agents(d, steps, seed=0, lr=1e-3):
    torch.manual_seed(seed); random.seed(seed)
    m = Aether(d=d).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        si, cn, ans, _ = _train.batch(BATCH)
        loss = F.cross_entropy(m(si, cn, _snr()), ans)
        opt.zero_grad(); loss.backward(); opt.step()
    return m, time.time() - t0


def digital_cliff(snr_db, d, ceiling, n_classes=10, chance=1.0 / K):
    """Classical baseline: send the target's class label (log2(C) bits) over d real
    channel uses at Shannon capacity. Perfect (=ceiling) above threshold, else outage."""
    bits_needed = math.log2(n_classes)
    cap = d * 0.5 * math.log2(1 + 10 ** (snr_db / 10.0))
    return ceiling if cap >= bits_needed else chance


def main():
    results = {"K": K, "chance": 1.0 / K, "snr_grid": SNR_GRID, "noise_range": NOISE_RANGE}

    # ---------- E1: bandwidth sweep ----------
    print("E1 bandwidth sweep")
    dims = [1, 2, 4, 8, 16, 32]
    sweep = {}
    for d in dims:
        m, secs = train_agents(d, steps=2000, seed=0)
        acc_clean = evaluate(m, 12.0)
        acc_noisy = evaluate(m, 0.0)
        sweep[d] = {"acc_clean": acc_clean, "acc_noisy": acc_noisy, "params": count_params(m)}
        print(f"  d={d:2d}  acc@12dB {acc_clean:.3f}  acc@0dB {acc_noisy:.3f}  ({secs:.0f}s)")
    results["bandwidth"] = sweep

    # ---------- E2: main model + noise curve vs digital cliff ----------
    print("E2 main model (d=8) noise curve")
    D_MAIN = 8
    main_m, secs = train_agents(D_MAIN, steps=3200, seed=0)
    print(f"  trained main d={D_MAIN} in {secs:.0f}s")
    analog = [evaluate(main_m, s) for s in SNR_GRID]
    ceiling = max(analog)
    digital = [digital_cliff(s, D_MAIN, ceiling) for s in SNR_GRID]
    results["main_d"] = D_MAIN
    results["noise_curve"] = {"analog": analog, "digital": digital, "ceiling": ceiling}
    for s, a, dg in zip(SNR_GRID, analog, digital):
        print(f"  {s:3d}dB  analog {a:.3f}  digital {dg:.3f}")

    # ---------- E3: desync ----------
    print("E3 desync")
    other, _ = train_agents(D_MAIN, steps=3200, seed=1)
    # cross-wire: sender from main, receiver from other
    class Crossed(torch.nn.Module):
        def __init__(s, se, re): super().__init__(); s.sender = se.sender; s.receiver = re.receiver
        def forward(s, img, cand, snr):
            from channel import awgn
            return s.receiver(awgn(s.sender(img), snr), cand)
    crossed = Crossed(main_m, other)
    matched_acc = evaluate(main_m, 16.0)
    crossed_acc = evaluate(crossed, 16.0)
    results["desync"] = {"matched": matched_acc, "crossed": crossed_acc, "snr": 16.0}
    print(f"  matched {matched_acc:.3f}   crossed {crossed_acc:.3f}   chance {1.0/K:.3f}")

    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print("wrote results.json")


if __name__ == "__main__":
    main()
