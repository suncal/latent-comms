"""
A standard wireless baseband channel — the kind a comms engineer would recognize.

Symbols are complex (I/Q). The transmitted vector of 2m reals is packed into m
complex symbols and power-normalized to unit average symbol power. Then:

  y = h ⊙ s + n + j_band

  * h  — block fading gain, Rayleigh or Rician(K), constant over a coherence
         block, E[|h|²] = 1 (a standard normalization).
  * n  — complex AWGN, E[|n|²] = N0 = 10^(-SNR/10).
  * j  — partial-band noise jamming: the jammer concentrates its total power over
         a fraction rho of the band, so the per-symbol jammer power in that band
         is J/rho. Smaller rho => a taller, narrower spike; the worst-case rho is
         the classic partial-band jamming attack.

The receiver has perfect channel state information (h), the standard coherent
assumption. Both y and h are handed to the decoder as features so it can perform
its own (learned) equalization — including the noise-amplifying reality of
dividing by a deeply-faded h.
"""

import math
import torch


def pack(z):
    """(B, 2m) real -> (B, m, 2) complex-as-2vector (first half real, second imag)."""
    m = z.shape[1] // 2
    return torch.stack([z[:, :m], z[:, m:]], dim=-1)


def power_normalize(s):
    """Scale each sample so mean complex-symbol power E[|s|^2] == 1."""
    p = (s ** 2).sum(-1).mean(dim=1, keepdim=True).clamp_min(1e-8).sqrt()  # (B,1)
    return s / p.unsqueeze(-1)


def cmul(a, b):
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)


def block_fading(B, m, block_len, device, kind="rayleigh", K=4.0):
    nb = math.ceil(m / block_len)
    scat = torch.randn(B, nb, 2, device=device) / math.sqrt(2.0)  # CN(0,1), E|.|^2=1
    if kind == "rayleigh":
        h = scat
    else:  # rician with K-factor (dB-free linear K)
        los = torch.zeros(B, nb, 2, device=device)
        los[..., 0] = math.sqrt(K / (K + 1.0))
        h = los + math.sqrt(1.0 / (K + 1.0)) * scat
    h = h.repeat_interleave(block_len, dim=1)[:, :m, :]
    return h  # (B, m, 2), E[|h|^2] = 1


def awgn(y, snr_db):
    N0 = 10.0 ** (-snr_db / 10.0)
    n = torch.randn_like(y) * math.sqrt(N0 / 2.0)  # per real dim
    return y + n, N0


def partial_band_jam(y, rho, jsr_db, start=None):
    """Add partial-band noise jamming; return (y, per-symbol jammer power vector)."""
    B, m, _ = y.shape
    jpow = torch.zeros(B, m, device=y.device)
    if rho <= 0 or jsr_db is None:
        return y, jpow
    J = 10.0 ** (jsr_db / 10.0)          # total jammer-to-signal ratio (per-symbol avg)
    band = max(1, int(round(rho * m)))
    per_sym = J / (band / m)             # concentrate: power scales by 1/rho
    if start is None:
        start = torch.randint(0, m - band + 1, (B,), device=y.device)
    idx = torch.arange(m, device=y.device).unsqueeze(0)
    in_band = (idx >= start.unsqueeze(1)) & (idx < (start.unsqueeze(1) + band))
    jpow = in_band.float() * per_sym
    jn = torch.randn_like(y) * (jpow.unsqueeze(-1) / 2.0).sqrt()
    return y + jn, jpow


def transmit(z, snr_db, kind="rayleigh", K=4.0, block_len=8, jam_rho=0.0, jsr_db=None):
    """Full channel. Returns decoder features [yr,yi,hr,hi] (B,m,4) and diagnostics."""
    s = power_normalize(pack(z))
    B, m, _ = s.shape
    h = block_fading(B, m, block_len, z.device, kind, K)
    y = cmul(h, s)
    y, N0 = awgn(y, snr_db)
    y, jpow = partial_band_jam(y, jam_rho, jsr_db)
    feats = torch.cat([y, h], dim=-1)          # (B, m, 4)
    # per-symbol SINR for the digital-baseline capacity calculation
    hmag2 = (h ** 2).sum(-1)                    # |h|^2
    sinr = hmag2 / (N0 + jpow + 1e-9)
    return feats, sinr
