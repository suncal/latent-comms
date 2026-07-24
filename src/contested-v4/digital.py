"""
Honest classical baseline: transform coding + a capacity-achieving code with
outage over the fading/jamming channel.

Source coding: per-channel 2-D DCT of the RGB image, uniform quantization, charged
the ideal per-frequency entropy (generous). This gives a rate-distortion curve
bits -> PSNR for the imagery.

Channel coding: with perfect receiver CSI, a capacity-achieving code carries up to
the *instantaneous* mutual information  I = sum_i log2(1 + SINR_i)  bits per block
realization. A practical fixed-rate system targets rate R: if I >= R the block
decodes perfectly (image at the R-bit quality); if I < R the block is in OUTAGE
and is lost (receiver falls back to the mean image). We let the digital system
pick its best R per condition — the strongest honest classical baseline.
"""

import numpy as np

N = 32


def _dct_matrix(n):
    idx = np.arange(n)
    D = np.cos(np.pi * (2 * idx + 1) * idx.reshape(-1, 1) / (2 * n))
    D[0, :] *= 1.0 / np.sqrt(n)
    D[1:, :] *= np.sqrt(2.0 / n)
    return D


_D = _dct_matrix(N)


def _dct2(a):    # a: (...,N,N)
    return _D @ a @ _D.T


def _idct2(c):
    return _D.T @ c @ _D


def rate_distortion_table(imgs, deltas):
    """imgs: (M,3,32,32) in [0,1]. Returns list of {bits, psnr} over quant steps."""
    imgs = imgs.astype(np.float64)
    M = imgs.shape[0]
    coeffs = np.stack([_dct2(imgs[:, c]) for c in range(3)], axis=1)  # (M,3,32,32)
    table = []
    for d in deltas:
        q = np.round(coeffs / d)
        bits = 0.0
        qf = q.reshape(M, -1)
        for p in range(qf.shape[1]):
            _, counts = np.unique(qf[:, p], return_counts=True)
            pr = counts / counts.sum()
            bits += float(-(pr * np.log2(pr)).sum())
        recon = np.stack([_idct2(q[:, c] * d) for c in range(3)], axis=1)
        recon = np.clip(recon, 0, 1)
        mse = np.mean((recon - imgs) ** 2)
        table.append({"delta": float(d), "bits": bits,
                      "psnr": float(10 * np.log10(1.0 / (mse + 1e-12)))})
    return table


def outage_psnr(imgs):
    imgs = imgs.astype(np.float64)
    mean_img = imgs.mean(axis=0, keepdims=True)
    mse = np.mean((imgs - mean_img) ** 2)
    return float(10 * np.log10(1.0 / (mse + 1e-12)))


def digital_expected_psnr(table, outage, inst_bits, best_over_rates=True):
    """inst_bits: array of realized capacities (bits) over many channel draws.
    Digital picks a fixed rate R; PSNR(R) = quality(R) * P(I>=R) + outage * P(I<R).
    Return the best achievable expected PSNR over R (the strongest fixed-rate system)."""
    inst_bits = np.asarray(inst_bits)
    best = outage
    for row in table:
        R = row["bits"]
        p_ok = float(np.mean(inst_bits >= R))
        exp_psnr = row["psnr"] * p_ok + outage * (1 - p_ok)
        if exp_psnr > best:
            best = exp_psnr
    return best
