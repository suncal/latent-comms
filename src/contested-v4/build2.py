"""
Build 2 — the moat: capabilities a digital link fundamentally cannot have.

1. Below the capacity floor ("the dead zone"): at very low SNR a digital system
   cannot fit even a minimal frame -> total outage (blank). The analog JSCC has no
   floor; it keeps delivering a usable image.
2. No transmitter CSI: rate-adaptive digital needs closed-loop channel knowledge to
   pick its rate. On one-way / broadcast / deep-space links you don't have it, so
   digital runs a FIXED rate and outages whenever the channel dips. JSCC needs no
   transmitter CSI at all.

Reuses the trained analog-128 model; no new training. Renders a dead-zone montage.
"""

import os
import json
import numpy as np
import torch
from PIL import Image

from data import load
from model import AdaptiveJSCC, psnr
import digital as dg
import hybrid as hy

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"
LOW_GRID = [-12, -10, -8, -6, -4, -2, 0, 3, 6]


def load_model(fname):
    ck = torch.load(os.path.join(HERE, fname), map_location=DEVICE)
    m = AdaptiveJSCC(m=ck["m"]).to(DEVICE); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


@torch.no_grad()
def jscc_psnr(model, x, snr, draws=3):
    return float(np.mean([psnr(x, model(x, snr_db=snr, kind="rayleigh")[0]).mean().item() for _ in range(draws)]))


def digital_fixed_psnr(table, outage, inst_bits, R_fixed):
    row = min(table, key=lambda r: abs(r["bits"] - R_fixed))
    p = float(np.mean(np.asarray(inst_bits) >= R_fixed))
    return row["psnr"] * p + outage * (1 - p), p


def montage(model, disp, table, outage, snr, fname, scale=3, pad=1):
    with torch.no_grad():
        rec = model(disp, snr_db=snr, kind="rayleigh")[0].clamp(0, 1).cpu().numpy()
    orig = disp.cpu().numpy()
    mean_img = load("test")[0][:1500].numpy().mean(0)   # dataset mean = digital outage output
    n = disp.shape[0]
    dig = np.stack([mean_img] * n)                        # digital in the dead zone: blank
    rows = [orig, rec, dig]
    H = 32 * scale
    out = np.full((3 * (H + pad) + pad, n * (32 * scale + pad) + pad, 3), 40, np.uint8)
    for ri, imgs in enumerate(rows):
        for ci in range(n):
            a = (np.clip(imgs[ci].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            a = np.kron(a, np.ones((scale, scale, 1), np.uint8))
            y0 = pad + ri * (H + pad); x0 = pad + ci * (32 * scale + pad)
            out[y0:y0 + H, x0:x0 + 32 * scale] = a
    Image.fromarray(out).save(os.path.join(HERE, fname))


def main():
    xva, _ = load("test", device=DEVICE)
    model = load_model("model_analog128.pt")
    table = dg.rate_distortion_table(xva[:1500].numpy(), np.geomspace(0.01, 3.0, 36))
    outage = dg.outage_psnr(xva.numpy())

    # fixed-rate digital designed for a 6 dB average channel (no TX CSI)
    ib6 = hy.inst_bits_kd(128, snr_db=6, kind="rayleigh")
    R_fixed = float(np.median(ib6))

    js, d_adapt, d_fixed, d_deliver = [], [], [], []
    for snr in LOW_GRID:
        js.append(jscc_psnr(model, xva, snr))
        ib = hy.inst_bits_kd(128, snr_db=snr, kind="rayleigh")
        d_adapt.append(dg.digital_expected_psnr(table, outage, ib))
        pf, p = digital_fixed_psnr(table, outage, ib, R_fixed)
        d_fixed.append(pf); d_deliver.append(p)
        print(f"  {snr:3d}dB  JSCC {js[-1]:5.1f}  digital(adapt) {d_adapt[-1]:5.1f}  "
              f"digital(fixed,noCSI) {pf:5.1f} [{p*100:3.0f}% delivered]")

    # dead zone: SNR where adaptive digital is within 1 dB of blank
    dead = [s for s, d in zip(LOW_GRID, d_adapt) if d <= outage + 1.0]
    montage(model, xva[:8], table, outage, -8, "deadzone.png")

    results = {"grid": LOW_GRID, "jscc": js, "digital_adaptive": d_adapt,
               "digital_fixed_noCSI": d_fixed, "delivered_frac": d_deliver,
               "outage_blank": outage, "dead_zone_snr": dead}
    json.dump(results, open(os.path.join(HERE, "build2_results.json"), "w"), indent=2)
    print(f"\nblank-frame floor = {outage:.1f} dB; digital dead zone (<=floor+1): {dead} dB")
    print("wrote build2_results.json + deadzone.png")


if __name__ == "__main__":
    main()
