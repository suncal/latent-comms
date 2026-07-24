"""
Classical digital baseline: transform coding + IDEAL entropy coding + Shannon
capacity. This is the "separation" architecture Deep JSCC is measured against, and
it is deliberately generous so the comparison is honest:

  * Source coding: 2-D DCT, uniform quantization, then charged the *per-frequency
    empirical entropy* (an ideal entropy coder — better than any real codec).
  * Channel coding: assumed capacity-achieving. With k real channel uses at a given
    SNR, it can deliver up to C = k * 0.5*log2(1+SNR) error-free bits — and above
    that threshold the bits arrive perfectly; below it, nothing decodes (outage).

The result is the classic digital "cliff": flawless while capacity suffices,
catastrophic the instant it doesn't.
"""

import numpy as np

N = 28


def _dct_matrix(n):
    idx = np.arange(n)
    D = np.cos(np.pi * (2 * idx + 1) * idx.reshape(-1, 1) / (2 * n))
    D[0, :] *= 1.0 / np.sqrt(n)
    D[1:, :] *= np.sqrt(2.0 / n)
    return D  # orthonormal: D @ D.T == I


_D = _dct_matrix(N)


def dct2(imgs):
    # imgs: (M, 28, 28) -> coeffs same shape
    return _D @ imgs @ _D.T


def idct2(coeffs):
    return _D.T @ coeffs @ _D


def rate_distortion_table(imgs, deltas):
    """For each quantization step delta, return (bits_per_image, psnr).

    bits_per_image = sum over the 784 DCT positions of that position's empirical
    entropy across the image set (ideal per-frequency entropy coder).
    """
    imgs = imgs.astype(np.float64)
    coeffs = dct2(imgs)  # (M,28,28)
    M = imgs.shape[0]
    table = []
    for d in deltas:
        q = np.round(coeffs / d)
        # ---- rate: per-position entropy summed over positions ----
        qf = q.reshape(M, -1)  # (M, 784)
        bits = 0.0
        for p in range(qf.shape[1]):
            vals, counts = np.unique(qf[:, p], return_counts=True)
            pr = counts / counts.sum()
            bits += float(-(pr * np.log2(pr)).sum())
        # ---- distortion ----
        recon = idct2(q * d)
        recon = np.clip(recon, 0.0, 1.0)
        mse = np.mean((recon - imgs) ** 2)
        psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))
        table.append({"delta": float(d), "bits": bits, "psnr": float(psnr)})
    return table


def outage_psnr(imgs):
    """PSNR when no bits get through: receiver outputs the dataset mean image."""
    imgs = imgs.astype(np.float64)
    mean_img = imgs.mean(axis=0, keepdims=True)
    mse = np.mean((imgs - mean_img) ** 2)
    return 10.0 * np.log10(1.0 / (mse + 1e-12))


def digital_psnr_at_budget(table, outage, bits_budget):
    """Best achievable PSNR given a hard bit budget (else outage)."""
    feasible = [row for row in table if row["bits"] <= bits_budget]
    if not feasible:
        return outage
    return max(row["psnr"] for row in feasible)
