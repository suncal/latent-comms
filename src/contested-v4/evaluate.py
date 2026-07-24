"""
Three-way comparison across SNR (Rayleigh), all using 128 complex channel uses:
  pure analog-128   (v3 adaptive model)      — graceful, no cliff, saturates
  pure digital-128  (capacity + outage)      — wins high SNR, cliffs/outages low SNR
  hybrid 64+64      (analog floor + digital residual boost)  — the dominance attempt

Goal: does the hybrid track the upper envelope — near digital high, near analog low —
so a single system is never far from best and never falls off a cliff?
"""

import os
import json
import numpy as np
import torch

from data import load
from model import AdaptiveJSCC, psnr
import digital as dg
import hybrid as hy

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"
SNR_GRID = [-2, 0, 2, 4, 6, 8, 10, 12, 15, 18]
KA, KD = 64, 64
DELTAS = np.geomspace(0.005, 3.0, 30)


def load_model(fname):
    ck = torch.load(os.path.join(HERE, fname), map_location=DEVICE)
    m = AdaptiveJSCC(m=ck["m"]).to(DEVICE); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


@torch.no_grad()
def analog_recon(model, x, snr, draws=2):
    ps, imgs = [], None
    for _ in range(draws):
        out, _ = model(x, snr_db=snr, kind="rayleigh")
        ps.append(psnr(x, out).mean().item()); imgs = out
    return float(np.mean(ps)), imgs.numpy()


def main():
    xva, _ = load("test", device=DEVICE)
    xs = xva[:800]                          # subset for the RD-heavy hybrid eval
    a128 = load_model("model_analog128.pt")
    a64 = load_model("model_analog64.pt")

    print("full-image RD table (pure digital-128)…")
    table_full = dg.rate_distortion_table(xva[:1500].numpy(), np.geomspace(0.01, 3.0, 36))
    outage = dg.outage_psnr(xva.numpy())

    pa, pd, hyb = [], [], []
    for snr in SNR_GRID:
        # pure analog-128
        p_analog128 = analog_recon(a128, xva, snr, draws=3)[0]
        pa.append(p_analog128)
        # pure digital-128
        ib128 = hy.inst_bits_kd(128, snr_db=snr, kind="rayleigh")
        pd.append(dg.digital_expected_psnr(table_full, outage, ib128))
        # hybrid: analog-64 floor + digital-64 residual
        p_a64, x_a = analog_recon(a64, xs, snr, draws=1)
        res_tab = hy.residual_rd_table(xs.numpy(), x_a, DELTAS)
        ib64 = hy.inst_bits_kd(KD, snr_db=snr, kind="rayleigh")
        hyb.append(hy.hybrid_expected_psnr(p_a64, res_tab, ib64))
        print(f"  {snr:3d}dB  analog128 {pa[-1]:5.1f}  digital128 {pd[-1]:5.1f}  hybrid {hyb[-1]:5.1f}")

    env = [max(a, d) for a, d in zip(pa, pd)]
    gap_closed = np.mean([(h - a) / (e - a + 1e-9) for h, a, e in zip(hyb, pa, env) if e > a])
    results = {"snr_grid": SNR_GRID, "pure_analog128": pa, "pure_digital128": pd,
               "hybrid": hyb, "envelope": env, "ka": KA, "kd": KD}
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print(f"\nhybrid recovers ~{gap_closed*100:.0f}% of the analog->best-of-both gap on average")
    print("wrote results.json")


if __name__ == "__main__":
    main()
