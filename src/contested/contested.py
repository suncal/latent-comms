"""
A contested / deep-space channel: the impairments real links face, not just AWGN.

Applied to the k transmitted symbols (B, k):
  * block fade   — the link fades in patches (multipath / atmospheric / pointing
                   loss): split the symbols into blocks, scale each by a random
                   gain, some blocks deeply faded toward zero.
  * burst erase  — a contiguous span of symbols is wiped out (a jammer parks on a
                   sub-band, or an interference burst): set that span to zero.
  * AWGN         — thermal noise, swept down to very low SNR.

The order matters: fade and jam happen in the air, then the receiver's noise adds
on top. SNR is defined at the transmitter (unit power), so fades and erasures cut
the *received* signal further — exactly the compounding stress of a contested link.
"""

import torch
from channel import awgn


def block_fade(z, depth, n_blocks=8, gen=None):
    """Per-block random gain in [1-depth, 1]; a few blocks driven deeply toward 0."""
    if depth <= 0:
        return z
    B, k = z.shape
    bl = k // n_blocks
    g = 1.0 - depth * torch.rand(B, n_blocks, device=z.device)
    # randomly deep-fade ~a third of the blocks toward zero
    deep = (torch.rand(B, n_blocks, device=z.device) < 0.3).float()
    g = g * (1 - deep) + deep * (0.05 * torch.rand(B, n_blocks, device=z.device))
    g = g.repeat_interleave(bl, dim=1)
    if g.shape[1] < k:                       # pad remainder
        g = torch.cat([g, g[:, -1:].expand(B, k - g.shape[1])], dim=1)
    return z * g


def burst_erase(z, frac, gen=None):
    """Zero a contiguous span covering `frac` of the symbols, random start per row."""
    if frac <= 0:
        return z
    B, k = z.shape
    L = int(round(frac * k))
    if L <= 0:
        return z
    start = torch.randint(0, k - L + 1, (B, 1), device=z.device)
    idx = torch.arange(k, device=z.device).unsqueeze(0)
    keep = ~((idx >= start) & (idx < start + L))
    return z * keep.float()


def contested(z, snr_db, erase_frac=0.0, fade_depth=0.0):
    z = block_fade(z, fade_depth)
    z = burst_erase(z, erase_frac)
    return awgn(z, snr_db)


def sample_train_channel():
    """Draw a random contested condition for one training batch."""
    import random
    snr = random.uniform(-4.0, 14.0)
    erase = random.uniform(0.0, 0.45) if random.random() < 0.7 else 0.0
    fade = random.uniform(0.0, 0.6) if random.random() < 0.7 else 0.0
    return snr, erase, fade
