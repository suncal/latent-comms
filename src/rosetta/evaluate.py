"""Evaluate interoperability: the 3x3 sender->receiver PSNR matrix, before vs after Rosetta."""

import os
import json
import numpy as np
import torch

from train_rosetta import load_bases, DEVICE, SEEDS
from rosetta import Rosetta
from data import load_fashion
from model import psnr
from channel import awgn

HERE = os.path.dirname(os.path.abspath(__file__))
SNRS = [0, 4, 8, 12]


def main():
    bases = load_bases()
    ros = Rosetta(bases).to(DEVICE)
    ck = torch.load(os.path.join(HERE, "rosetta_adapters.pt"), map_location=DEVICE)
    ros.to_ref.load_state_dict(ck["to_ref"]); ros.from_ref.load_state_dict(ck["from_ref"])
    ros.eval()
    x = load_fashion("test", limit=2000, device=DEVICE)
    N = ros.n

    @torch.no_grad()
    def matrices(snr):
        before = np.zeros((N, N)); after = np.zeros((N, N))
        for i in range(N):
            z = bases[i].enc(x)
            r_rx = ros.transmit(x, i, snr)
            for j in range(N):
                before[i, j] = psnr(x, bases[j].dec(awgn(z, snr))).mean().item()
                after[i, j] = psnr(x, ros.receive(r_rx, j)).mean().item()
        return before, after

    b10, a10 = matrices(10.0)
    # off-diagonal (cross-vendor) averages across SNR
    offmask = ~np.eye(N, dtype=bool)
    curve = {"snr": SNRS, "before_cross": [], "after_cross": [], "after_matched": []}
    for s in SNRS:
        b, a = matrices(s)
        curve["before_cross"].append(float(b[offmask].mean()))
        curve["after_cross"].append(float(a[offmask].mean()))
        curve["after_matched"].append(float(np.diag(a).mean()))

    results = {"seeds": SEEDS, "snr_eval": 10.0,
               "before_matrix": b10.round(1).tolist(), "after_matrix": a10.round(1).tolist(),
               "before_cross_avg": float(b10[offmask].mean()), "after_cross_avg": float(a10[offmask].mean()),
               "matched_avg": float(np.diag(b10).mean()), "curve": curve,
               "adapter_params": int(sum(p.numel() for p in ros.parameters() if p.requires_grad)),
               "base_params_each": int(sum(p.numel() for p in bases[0].parameters()))}
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)

    print("BEFORE (raw cross-model) @10dB:"); print(b10.round(1))
    print("AFTER (through Rosetta) @10dB:"); print(a10.round(1))
    print(f"\ncross-vendor avg: {results['before_cross_avg']:.1f} dB -> {results['after_cross_avg']:.1f} dB "
          f"(matched reference {results['matched_avg']:.1f} dB)")
    print(f"adapters: {results['adapter_params']:,} params vs {results['base_params_each']:,}/base model")
    print("wrote results.json")


if __name__ == "__main__":
    main()
