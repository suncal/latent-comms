"""
Evaluate under contested conditions and build the report data + montages.

E1  PSNR vs burst-jamming fraction (at low SNR): contested model vs AWGN-only
    model vs classical digital.
E2  PSNR vs SNR (at fixed 25% jamming): contested vs digital.
E3  Visual montage: images under rising jamming — recognizable vs blank.
"""

import os
import json
import numpy as np
import torch
from PIL import Image

from data import load_fashion
from model import LatentRadio, psnr
from contested import contested
import digital as dg
from channel import channel_capacity_bits

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"
K = 128
JAM_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SNR_GRID = [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12]
DRAWS = 4


def load_model(mode):
    ck = torch.load(os.path.join(HERE, f"model_{mode}.pt"), map_location=DEVICE)
    m = LatentRadio(k=ck["k"]).to(DEVICE); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


@torch.no_grad()
def mean_psnr(model, x, snr, erase, fade):
    vals = []
    for _ in range(DRAWS):
        rec = model.dec(contested(model.enc(x), snr, erase, fade))
        vals.append(psnr(x, rec).mean().item())
    return float(np.mean(vals))


def digital_curve_jam(table, outage, snr, jam_grid):
    """Classical digital with ideal interleaving: erasing fraction f cuts the
    delivered capacity to (1-f)*capacity. Generous to digital (no burst cliff)."""
    out = []
    for f in jam_grid:
        cap = (1 - f) * channel_capacity_bits(K, snr)
        out.append(dg.digital_psnr_at_budget(table, outage, cap))
    return out


def digital_curve_snr(table, outage, jam, snr_grid):
    out = []
    for s in snr_grid:
        cap = (1 - jam) * channel_capacity_bits(K, s)
        out.append(dg.digital_psnr_at_budget(table, outage, cap))
    return out


def montage(model_c, model_a, disp, table, outage, snr, jams, fname, scale=4, pad=1):
    imgs = disp[:, 0].numpy().astype(np.float64)
    n = imgs.shape[0]
    rows = [("original", [imgs[i] for i in range(n)])]
    with torch.no_grad():
        for j in jams:
            r = model_c.dec(contested(model_c.enc(disp), snr, j, 0.0))[:, 0].numpy()
            rows.append((f"contested-trained @ {int(j*100)}% jam", [r[i] for i in range(n)]))
        # one row for AWGN-only model under heavy jam
        j = jams[-1]
        r = model_a.dec(contested(model_a.enc(disp), snr, j, 0.0))[:, 0].numpy()
        rows.append((f"AWGN-only model @ {int(j*100)}% jam", [r[i] for i in range(n)]))
    # digital under heavy jam
    cap = (1 - jams[-1]) * channel_capacity_bits(K, snr)
    feas = [row for row in table if row["bits"] <= cap]
    if feas:
        d = max(feas, key=lambda r: r["psnr"])["delta"]
        q = np.round(dg.dct2(imgs) / d) * d
        drec = np.clip(dg.idct2(q), 0, 1)
        drow = [drec[i] for i in range(n)]
    else:
        drow = [imgs.mean(axis=0) for _ in range(n)]
    rows.append((f"classical digital @ {int(jams[-1]*100)}% jam", drow))

    cell = 28 * scale
    H = len(rows) * cell + (len(rows) + 1) * pad
    W = n * cell + (n + 1) * pad
    canvas = np.full((H, W), 40, np.uint8)
    for ri, (_, cells) in enumerate(rows):
        for ci, g in enumerate(cells):
            a = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            a = np.kron(a, np.ones((scale, scale), np.uint8))
            y0 = pad + ri * (cell + pad); x0 = pad + ci * (cell + pad)
            canvas[y0:y0 + cell, x0:x0 + cell] = a
    Image.fromarray(canvas).save(os.path.join(HERE, fname))
    return [r[0] for r in rows]


def main():
    xva = load_fashion("test", limit=2000, device=DEVICE)
    disp = load_fashion("test", limit=8, device=DEVICE)
    mc, ma = load_model("contested"), load_model("awgn_only")

    print("digital RD table…")
    imgs = xva[:, 0].numpy()
    table = dg.rate_distortion_table(imgs, np.geomspace(0.01, 3.0, 40))
    outage = dg.outage_psnr(imgs)

    LOW_SNR = 2.0
    print(f"E1 jamming sweep @ {LOW_SNR}dB")
    e1_contested = [mean_psnr(mc, xva, LOW_SNR, j, 0.0) for j in JAM_GRID]
    e1_awgn = [mean_psnr(ma, xva, LOW_SNR, j, 0.0) for j in JAM_GRID]
    e1_digital = digital_curve_jam(table, outage, LOW_SNR, JAM_GRID)

    JAM = 0.25
    print(f"E2 SNR sweep @ {int(JAM*100)}% jam")
    e2_contested = [mean_psnr(mc, xva, s, JAM, 0.0) for s in SNR_GRID]
    e2_digital = digital_curve_snr(table, outage, JAM, SNR_GRID)

    print("E3 montage")
    labels = montage(mc, ma, disp, table, outage, 4.0, [0.0, 0.25, 0.45], "contested_montage.png")

    results = {
        "K": K, "low_snr": LOW_SNR, "jam_fixed": JAM, "outage": outage,
        "jam_grid": JAM_GRID, "snr_grid": SNR_GRID,
        "e1": {"contested": e1_contested, "awgn_only": e1_awgn, "digital": e1_digital},
        "e2": {"contested": e2_contested, "digital": e2_digital},
        "montage_rows": labels,
    }
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print("\nE1 jam% : contested / awgn-only / digital")
    for j, a, b, c in zip(JAM_GRID, e1_contested, e1_awgn, e1_digital):
        print(f"  {int(j*100):3d}%   {a:5.1f} / {b:5.1f} / {c:5.1f}")
    print("wrote results.json + contested_montage.png")


if __name__ == "__main__":
    main()
