"""
Evaluate the JSCC transceiver against the outage-based digital baseline under
standard channel models, and build report data + a satellite montage.

E1  PSNR vs SNR under Rayleigh fading (no jam)     — JSCC vs digital.
E2  PSNR vs SNR under Rician K=6 (milder fading)   — JSCC vs digital.
E3  Worst-case partial-band jamming: sweep the jammer's band fraction rho at fixed
    JSR and SNR; the jammer picks the rho that hurts most. JSCC vs digital.
E4  Satellite montage under Rayleigh + jamming.
"""

import os
import json
import numpy as np
import torch

from data import load, CLASSES
from model import ContestedJSCC, psnr, M
from channel import transmit
import digital as dg

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"
SNR_GRID = [-2, 0, 2, 4, 6, 8, 10, 12, 15, 18]
RHO_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
DRAWS = 3


def load_model():
    ck = torch.load(os.path.join(HERE, "jscc.pt"), map_location=DEVICE)
    m = ContestedJSCC(m=ck["m"]).to(DEVICE); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


@torch.no_grad()
def jscc_psnr(model, x, **cond):
    vals = []
    for _ in range(DRAWS):
        out, _ = model(x, **cond)
        vals.append(psnr(x, out).mean().item())
    return float(np.mean(vals))


def inst_bits_dist(n=3000, **cond):
    """Distribution of instantaneous capacity (bits) for a channel condition."""
    z = torch.randn(n, 2 * M)
    _, sinr = transmit(z, **cond)          # (n, M)
    return torch.log2(1 + sinr).sum(1).numpy()


def digital_psnr(table, outage, **cond):
    return dg.digital_expected_psnr(table, outage, inst_bits_dist(**cond))


def montage(model, disp, table, outage, cond, fname, scale=3, pad=1):
    import numpy as np
    from PIL import Image
    x = disp
    n = x.shape[0]
    with torch.no_grad():
        rec = model(x, **cond)[0].clamp(0, 1).cpu().numpy()
    orig = x.cpu().numpy()
    # digital recon at its best feasible rate for this condition
    ib = inst_bits_dist(n=2000, **cond)
    # choose R that maximizes expected psnr, then render the images at that rate
    best = None
    for row in table:
        p_ok = float(np.mean(ib >= row["bits"]))
        e = row["psnr"] * p_ok + outage * (1 - p_ok)
        if best is None or e > best[0]:
            best = (e, row["delta"], p_ok)
    _, d, p_ok = best
    dimgs = orig.astype(np.float64)
    drec = np.zeros_like(dimgs)
    for c in range(3):
        q = np.round(dg._dct2(dimgs[:, c]) / d) * d
        drec[:, c] = np.clip(dg._idct2(q), 0, 1)
    # apply outage: a fraction (1-p_ok) of blocks are lost -> mean image
    mean_img = dimgs.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(0)
    lost = rng.random(n) > p_ok
    drec[lost] = mean_img[0]

    def grid(rows):
        H = 32 * scale
        cellH = H
        out = np.full((len(rows) * (cellH + pad) + pad, n * (32 * scale + pad) + pad, 3), 40, np.uint8)
        for ri, imgs in enumerate(rows):
            for ci in range(n):
                a = (np.clip(imgs[ci].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                a = np.kron(a, np.ones((scale, scale, 1), np.uint8))
                y0 = pad + ri * (cellH + pad); x0 = pad + ci * (32 * scale + pad)
                out[y0:y0 + cellH, x0:x0 + 32 * scale] = a
        return out
    canvas = grid([orig, rec, drec])
    Image.fromarray(canvas).save(os.path.join(HERE, fname))
    return round(p_ok, 3)


def main():
    xva, _ = load("test", device=DEVICE)
    model = load_model()
    print("digital RD table (satellite imagery)…")
    table = dg.rate_distortion_table(xva[:1500].numpy(), np.geomspace(0.01, 3.0, 36))
    outage = dg.outage_psnr(xva.numpy())

    print("E1 Rayleigh SNR sweep")
    e1_j = [jscc_psnr(model, xva, snr_db=s, kind="rayleigh") for s in SNR_GRID]
    e1_d = [digital_psnr(table, outage, snr_db=s, kind="rayleigh") for s in SNR_GRID]

    print("E2 Rician K=6 SNR sweep")
    e2_j = [jscc_psnr(model, xva, snr_db=s, kind="rician", K=6.0) for s in SNR_GRID]
    e2_d = [digital_psnr(table, outage, snr_db=s, kind="rician", K=6.0) for s in SNR_GRID]

    print("E3 worst-case partial-band jamming @ 10dB Rayleigh, JSR=10dB")
    e3_j = [jscc_psnr(model, xva, snr_db=10, kind="rayleigh", jam_rho=r, jsr_db=10.0) for r in RHO_GRID]
    e3_d = [digital_psnr(table, outage, snr_db=10, kind="rayleigh", jam_rho=r, jsr_db=10.0) for r in RHO_GRID]

    print("E4 montage")
    disp = xva[:8]
    p_ok = montage(model, disp, table, outage,
                   dict(snr_db=8, kind="rayleigh", jam_rho=0.3, jsr_db=8.0), "sat_montage.png")

    results = {
        "M": M, "snr_grid": SNR_GRID, "rho_grid": RHO_GRID, "outage": outage,
        "e1": {"jscc": e1_j, "digital": e1_d},
        "e2": {"jscc": e2_j, "digital": e2_d},
        "e3": {"jscc": e3_j, "digital": e3_d, "snr": 10, "jsr": 10},
        "montage_digital_pok": p_ok,
    }
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print("\nE1 Rayleigh  SNR: jscc / digital")
    for s, j, d in zip(SNR_GRID, e1_j, e1_d): print(f"  {s:3d}dB  {j:5.1f} / {d:5.1f}")
    print("E3 jam rho: jscc / digital  (worst-case = digital min)")
    for r, j, d in zip(RHO_GRID, e3_j, e3_d): print(f"  rho={r:.2f}  {j:5.1f} / {d:5.1f}")
    print("wrote results.json + sat_montage.png")


if __name__ == "__main__":
    main()
