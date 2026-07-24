"""
Hybrid digital-analog evaluation.

The hybrid spends k_a complex uses on an analog JSCC layer (a graceful full-image
reconstruction, no cliff) and k_d uses on a digital layer that codes the RESIDUAL
(image minus analog reconstruction). Key property: when the digital layer outages,
the fallback is the graceful analog image — never a blank frame.

For a channel condition we compute the EXPECTED hybrid PSNR:
    max over residual-rate R of
        P(capacity_kd >= R) * PSNR(analog + decoded_residual@R)
      + P(capacity_kd <  R) * PSNR(analog alone)
The analog term is the no-cliff floor; the digital term is the good-channel boost.
"""

import numpy as np
import torch

import digital as dg
from channel import transmit


def inst_bits_kd(kd_complex, n=3000, **cond):
    """Instantaneous capacity (bits) available to the digital layer over kd uses."""
    z = torch.randn(n, 2 * kd_complex)
    _, sinr = transmit(z, **cond)
    return torch.log2(1 + sinr).sum(1).numpy()


def residual_rd_table(x, x_a, deltas):
    """x, x_a: (M,3,32,32). For each quant step on the residual, return
    {bits, psnr} where psnr is for clip(x_a + decoded_residual) vs x."""
    x = x.astype(np.float64); x_a = x_a.astype(np.float64)
    r = x - x_a
    M = x.shape[0]
    coeffs = np.stack([dg._dct2(r[:, c]) for c in range(3)], axis=1)
    table = []
    for d in deltas:
        q = np.round(coeffs / d)
        qf = q.reshape(M, -1)
        bits = 0.0
        for p in range(qf.shape[1]):
            _, counts = np.unique(qf[:, p], return_counts=True)
            pr = counts / counts.sum()
            bits += float(-(pr * np.log2(pr)).sum())
        r_hat = np.stack([dg._idct2(q[:, c] * d) for c in range(3)], axis=1)
        hyb = np.clip(x_a + r_hat, 0, 1)
        mse = np.mean((hyb - x) ** 2)
        table.append({"bits": bits, "psnr": float(10 * np.log10(1.0 / (mse + 1e-12)))})
    return table


def hybrid_expected_psnr(analog_psnr, res_table, inst_bits):
    inst_bits = np.asarray(inst_bits)
    best = analog_psnr  # worst case: digital never helps -> analog floor
    for row in res_table:
        p = float(np.mean(inst_bits >= row["bits"]))
        exp = row["psnr"] * p + analog_psnr * (1 - p)
        if exp > best:
            best = exp
    return best
