"""
Train Sigil and measure how gracefully it degrades under damage.

Two models, identical except for training:
  * channel-aware  — trained through the damage channel (the real Sigil).
  * no-channel     — trained on clean glyphs only (ablation: shows the joint
                     training is what buys robustness).

For each, sweep damage severity and report bit accuracy and exact-message
recovery. Also save example glyphs (clean + damaged) for the report.
Writes results.json, plus glyph PNGs.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import Encoder, Decoder, damage, bit_metrics, NBITS, GLYPH

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
STEPS = 3500
BATCH = 256
SEV_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def rand_bits(B):
    return (torch.rand(B, NBITS, device=DEVICE) > 0.5).float()


def train(channel_aware, seed=0):
    torch.manual_seed(seed)
    enc, dec = Encoder().to(DEVICE), Decoder().to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    for step in range(1, STEPS + 1):
        bits = rand_bits(BATCH)
        glyph = enc(bits)
        if channel_aware:
            sev = float(torch.empty(1).uniform_(0.0, 1.0).item())
            glyph_rx = damage(glyph, sev)
        else:
            glyph_rx = glyph  # trained on clean only
        logits = dec(glyph_rx)
        loss = F.binary_cross_entropy_with_logits(logits, bits)
        opt.zero_grad(); loss.backward(); opt.step()
    return enc, dec


@torch.no_grad()
def sweep(enc, dec, n=4000, draws=3):
    enc.eval(); dec.eval()
    bit_curve, msg_curve = [], []
    for sev in SEV_GRID:
        bas, mas = [], []
        for _ in range(draws):
            bits = rand_bits(n)
            rx = damage(enc(bits), sev)
            ba, ma = bit_metrics(dec(rx), bits)
            bas.append(ba); mas.append(ma)
        bit_curve.append(float(np.mean(bas)))
        msg_curve.append(float(np.mean(mas)))
    return bit_curve, msg_curve


def save_glyph_row(enc, bits, sevs, fname, scale=8):
    with torch.no_grad():
        glyph = enc(bits)
        cells = []
        for s in sevs:
            g = damage(glyph, s)[0, 0].cpu().numpy()
            cells.append(g)
    pad = 2
    cell = GLYPH * scale
    W = len(cells) * cell + (len(cells) + 1) * pad
    canvas = np.full((cell + 2 * pad, W), 30, np.uint8)
    for i, g in enumerate(cells):
        a = (np.clip(g, 0, 1) * 255).astype(np.uint8)
        a = np.kron(a, np.ones((scale, scale), np.uint8))
        x0 = pad + i * (cell + pad)
        canvas[pad:pad + cell, x0:x0 + cell] = a
    Image.fromarray(canvas).save(os.path.join(HERE, fname))


def main():
    print(f"device={DEVICE}  bits={NBITS}  glyph={GLYPH}x{GLYPH}")
    print("training channel-aware Sigil ...")
    enc, dec = train(channel_aware=True, seed=0)
    print("training no-channel ablation ...")
    enc0, dec0 = train(channel_aware=False, seed=0)

    print("sweeping damage ...")
    sig_bit, sig_msg = sweep(enc, dec)
    abl_bit, abl_msg = sweep(enc0, dec0)

    torch.manual_seed(7)
    demo_bits = rand_bits(1)
    save_glyph_row(enc, demo_bits, [0.0, 0.3, 0.5, 0.7, 0.9], "sigil_row.png")

    results = {
        "nbits": NBITS, "glyph": GLYPH, "sev_grid": SEV_GRID,
        "sigil": {"bit_acc": sig_bit, "msg_acc": sig_msg},
        "ablation": {"bit_acc": abl_bit, "msg_acc": abl_msg},
    }
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict()}, os.path.join(HERE, "sigil.pt"))
    print("\nsev   Sigil(bit/msg)   noChan(bit/msg)")
    for i, s in enumerate(SEV_GRID):
        print(f"{s:.1f}   {sig_bit[i]:.3f}/{sig_msg[i]:.3f}      {abl_bit[i]:.3f}/{abl_msg[i]:.3f}")
    print("wrote results.json, sigil.pt, sigil_row.png")


if __name__ == "__main__":
    main()
