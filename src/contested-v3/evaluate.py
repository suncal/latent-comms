"""
Evaluate the SNR-adaptive and reactive-hardened models.

A  Rate adaptivity: SNR sweep under Rayleigh — adaptive JSCC vs the v2 non-adaptive
   JSCC vs rate-adaptive digital. Does conditioning on SNR close the good-channel gap?
B  Reactive jamming: at fixed SNR, PSNR under no-jam / follower jammer / worst-case
   adversarial (PGD) jammer, for the standard-adaptive model vs the reactive-hardened
   model. Does adversarial training restore robustness?
"""

import os
import json
import numpy as np
import torch

from data import load
from model import AdaptiveJSCC, psnr, M
from channel import pack, power_normalize, cmul, block_fading, awgn, transmit
from jammers import follower_jam, adversarial_jam
import digital as dg

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"
SNR_GRID = [-2, 0, 2, 4, 6, 8, 10, 12, 15, 18]
DRAWS = 3


def load_model(tag):
    ck = torch.load(os.path.join(HERE, f"model_{tag}.pt"), map_location=DEVICE)
    m = AdaptiveJSCC(m=ck["m"]).to(DEVICE); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


@torch.no_grad()
def adaptive_psnr(model, x, snr, kind="rayleigh", K=4.0):
    vals = []
    for _ in range(DRAWS):
        out, _ = model(x, snr_db=snr, kind=kind, K=K)
        vals.append(psnr(x, out).mean().item())
    return float(np.mean(vals))


def digital_psnr(table, outage, **cond):
    z = torch.randn(3000, 2 * M)
    _, sinr = transmit(z, **cond)
    ib = torch.log2(1 + sinr).sum(1).numpy()
    return dg.digital_expected_psnr(table, outage, ib)


def jam_eval(model, x, snr, jsr, kind="rayleigh", K=4.0):
    """PSNR under: no jam / follower jam / worst-case adversarial jam."""
    outs = {}
    z, cond = model.encode(x, snr)
    s = power_normalize(pack(z)); B, m, _ = s.shape
    h = block_fading(B, m, 8, x.device, kind, K)
    y0 = cmul(h, s); y0, _ = awgn(y0, snr)
    with torch.no_grad():
        outs["none"] = psnr(x, model.dec(torch.cat([y0, h], -1), cond)).mean().item()
        yf = follower_jam(y0, s, jsr, rho=0.3)
        outs["follower"] = psnr(x, model.dec(torch.cat([yf, h], -1), cond)).mean().item()
    ya = adversarial_jam(model, cond, y0, h, x, jsr, steps=8)
    with torch.no_grad():
        outs["adversarial"] = psnr(x, model.dec(torch.cat([ya, h], -1), cond)).mean().item()
    return outs


def main():
    xva, _ = load("test", device=DEVICE)
    xj = xva[:800]
    adaptive = load_model("adaptive")
    robust = load_model("robust")
    v2 = json.load(open(os.path.join(HERE, "v2_results.json")))

    print("digital RD table…")
    table = dg.rate_distortion_table(xva[:1500].numpy(), np.geomspace(0.01, 3.0, 36))
    outage = dg.outage_psnr(xva.numpy())

    print("A: SNR sweep (rate adaptivity)")
    a_adaptive = [adaptive_psnr(adaptive, xva, s) for s in SNR_GRID]
    a_digital = [digital_psnr(table, outage, snr_db=s, kind="rayleigh") for s in SNR_GRID]
    a_v2 = v2["e1"]["jscc"]        # non-adaptive JSCC from v2 (same grid & channel)

    print("B: reactive jamming @ 8dB, JSR 10dB")
    JSR = 10.0
    b_adaptive = jam_eval(adaptive, xj, 8.0, JSR)
    b_robust = jam_eval(robust, xj, 8.0, JSR)
    print("   adaptive:", {k: round(v, 1) for k, v in b_adaptive.items()})
    print("   robust  :", {k: round(v, 1) for k, v in b_robust.items()})

    results = {
        "snr_grid": SNR_GRID,
        "A": {"adaptive": a_adaptive, "digital": a_digital, "nonadaptive_v2": a_v2},
        "B": {"adaptive": b_adaptive, "robust": b_robust, "snr": 8, "jsr": JSR},
    }
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print("\nA SNR: adaptive / non-adaptive(v2) / digital")
    for s, a, na, d in zip(SNR_GRID, a_adaptive, a_v2, a_digital):
        print(f"  {s:3d}dB  {a:5.1f} / {na:5.1f} / {d:5.1f}")
    print("wrote results.json")


if __name__ == "__main__":
    main()
