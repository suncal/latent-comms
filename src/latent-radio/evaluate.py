"""
Run the Latent Radio experiments and save results.json + reconstruction montages.

E1  PSNR vs channel SNR:  Latent Radio (graceful)  vs  digital baseline (cliff).
E2  Visual reconstructions at several SNRs for both systems.
E3a Failure — desync: encoder from model_0 + decoder from model_1 (two independently
    trained models). AI-native comms require both ends to share the SAME model.
E3b Failure — out-of-distribution: model trained on Fashion-MNIST, tested on digits.
"""

import os
import json
import numpy as np
import torch
from PIL import Image

from data import load_fashion, load_mnist_test
from model import LatentRadio, psnr
from channel import channel_capacity_bits
import digital as dg

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"  # eval is light; CPU keeps noise draws deterministic & simple
SNR_GRID = list(range(-6, 15, 2))         # dB
DISPLAY_SNRS = [-4, 2, 10]                 # for the image montages
NOISE_DRAWS = 4                            # average JSCC PSNR over this many draws
K = 128


def load_model(seed):
    ckpt = torch.load(os.path.join(HERE, "models", f"model_{seed}.pt"), map_location=DEVICE)
    m = LatentRadio(k=ckpt["k"]).to(DEVICE)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def jscc_psnr_curve(enc_model, dec_model, x):
    """Mean PSNR at each SNR, averaging over noise draws. enc/dec may differ (desync)."""
    out = []
    with torch.no_grad():
        for s in SNR_GRID:
            vals = []
            for _ in range(NOISE_DRAWS):
                z = enc_model.enc(x)
                from channel import awgn
                y = awgn(z, s)
                rec = dec_model.dec(y)
                vals.append(psnr(x, rec).mean().item())
            out.append(float(np.mean(vals)))
    return out


def digital_curve(table, outage):
    return [dg.digital_psnr_at_budget(table, outage, channel_capacity_bits(K, s)) for s in SNR_GRID]


def delta_for_budget(table, budget):
    feasible = [r for r in table if r["bits"] <= budget]
    if not feasible:
        return None
    return max(feasible, key=lambda r: r["psnr"])["delta"]


# ---------- montage helpers ----------
def to_uint8_grid(rows, scale=4, pad=1):
    """rows: list of lists of (28,28) float arrays -> single upscaled PIL image."""
    R, C = len(rows), len(rows[0])
    cell = 28 * scale
    H = R * cell + (R + 1) * pad
    W = C * cell + (C + 1) * pad
    canvas = np.full((H, W), 40, dtype=np.uint8)  # dark padding
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            a = np.clip(img, 0, 1)
            a = (a * 255).astype(np.uint8)
            a = np.kron(a, np.ones((scale, scale), dtype=np.uint8))
            y0 = pad + r * (cell + pad)
            x0 = pad + c * (cell + pad)
            canvas[y0:y0 + cell, x0:x0 + cell] = a
    return Image.fromarray(canvas, mode="L")


def build_montage(model, disp, table, outage, fname):
    """Rows: original, then JSCC@SNR, then digital@SNR for each display SNR."""
    imgs = disp[:, 0].numpy().astype(np.float64)
    rows, labels = [], []
    rows.append([imgs[i] for i in range(imgs.shape[0])]); labels.append("original")
    with torch.no_grad():
        for s in DISPLAY_SNRS:
            rec = model(disp, s)[:, 0].numpy()
            rows.append([rec[i] for i in range(rec.shape[0])]); labels.append(f"Latent Radio @ {s} dB")
    for s in DISPLAY_SNRS:
        budget = channel_capacity_bits(K, s)
        d = delta_for_budget(table, budget)
        if d is None:
            mean_img = imgs.mean(axis=0)
            row = [mean_img for _ in range(imgs.shape[0])]
        else:
            coeffs = dg.dct2(imgs)
            q = np.round(coeffs / d) * d
            recon = np.clip(dg.idct2(q), 0, 1)
            row = [recon[i] for i in range(recon.shape[0])]
        rows.append(row); labels.append(f"digital @ {s} dB")
    img = to_uint8_grid(rows)
    img.save(os.path.join(HERE, fname))
    return labels


def main():
    xva = load_fashion("test", limit=2000, device=DEVICE)
    disp = load_fashion("test", limit=8, device=DEVICE)  # 8 images for montages

    m0 = load_model(0)
    m1 = load_model(1)

    # digital baseline RD table on the same test set
    imgs_np = xva[:, 0].numpy()
    deltas = np.geomspace(0.01, 3.0, 40)
    print("building digital rate-distortion table ...")
    table = dg.rate_distortion_table(imgs_np, deltas)
    outage = dg.outage_psnr(imgs_np)

    print("E1: PSNR vs SNR ...")
    jscc = jscc_psnr_curve(m0, m0, xva)
    digital = digital_curve(table, outage)

    print("E3a: desync (enc0 + dec1) ...")
    desync = jscc_psnr_curve(m0, m1, xva)

    print("E3b: out-of-distribution (digits) ...")
    try:
        ood_x = load_mnist_test(limit=2000, device=DEVICE)
        ood = jscc_psnr_curve(m0, m0, ood_x)
    except Exception as e:
        print("  OOD skipped:", e)
        ood = None

    print("montages ...")
    labels = build_montage(m0, disp, table, outage, "recon.png")
    # desync montage: original, enc0+dec0, enc0+dec1 at one SNR
    with torch.no_grad():
        s = 10.0
        good = m0(disp, s)[:, 0].numpy()
        from channel import awgn
        z = m0.enc(disp); bad = m1.dec(awgn(z, s))[:, 0].numpy()
        og = disp[:, 0].numpy()
        rows = [[og[i] for i in range(8)],
                [good[i] for i in range(8)],
                [bad[i] for i in range(8)]]
    to_uint8_grid(rows).save(os.path.join(HERE, "desync.png"))

    results = {
        "k": K, "snr_grid": SNR_GRID, "display_snrs": DISPLAY_SNRS,
        "montage_row_labels": labels,
        "outage_psnr": outage,
        "capacity_bits": [channel_capacity_bits(K, s) for s in SNR_GRID],
        "curves": {
            "latent_radio": jscc,
            "digital": digital,
            "desync": desync,
            "ood_digits": ood,
        },
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results.json, recon.png, desync.png")

    print("\nSNR   LatentRadio  Digital   (dB PSNR)")
    for i, s in enumerate(SNR_GRID):
        print(f"{s:3d}dB    {jscc[i]:6.2f}    {digital[i]:6.2f}")


if __name__ == "__main__":
    main()
