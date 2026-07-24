"""The physical channel: power normalization + additive white Gaussian noise.

Convention: every real channel symbol carries average power 1, so the noise
variance is sigma^2 = 1 / SNR_linear and SNR(dB) = 10*log10(SNR_linear). This is
the standard normalization used in the Deep JSCC literature and it makes the
capacity comparison against the digital baseline (0.5*log2(1+SNR) bits per real
use) apples-to-apples: both systems get the SAME number of real channel uses.
"""

import torch


def snr_db_to_linear(snr_db):
    return 10.0 ** (snr_db / 10.0)


def power_normalize(z):
    """Scale each sample so its average power per real symbol == 1.
    z: (B, k) -> (B, k) with mean(x^2) == 1 per row."""
    k = z.shape[1]
    power = torch.sqrt((z ** 2).sum(dim=1, keepdim=True) / k + 1e-8)
    return z / power


def awgn(x, snr_db):
    """Add white Gaussian noise at the given SNR (signal power == 1 per symbol)."""
    snr = snr_db_to_linear(snr_db)
    sigma = (1.0 / snr) ** 0.5
    return x + sigma * torch.randn_like(x)


def channel_capacity_bits(k, snr_db):
    """Shannon capacity of k real AWGN uses at this SNR, in bits (generous
    upper bound the digital baseline is allowed to achieve)."""
    import math
    snr = snr_db_to_linear(snr_db)
    return k * 0.5 * math.log2(1.0 + snr)
