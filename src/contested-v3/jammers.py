"""
Reactive jammers — they adapt to the transmission, unlike a fixed partial-band jammer.

follower_jam: a sensing jammer that measures which symbols carry the most energy
  and concentrates its power on exactly those. Realistic and cheap.

adversarial_jam: the worst-case jammer. Given the actual model and a power budget,
  it runs projected gradient ascent to find the jamming perturbation on the received
  symbols that maximizes the decoder's reconstruction error. An upper bound on how
  much damage a smart jammer with the same power could do (assumes it knows the model
  and target — a pessimistic, honest stress test).

Both are power-constrained to a jammer-to-signal ratio (JSR); signal power is 1 per
complex symbol, so a total jam power budget of J*m is spread as the jammer sees fit.
"""

import math
import torch

from channel import pack, power_normalize, cmul, block_fading, awgn


def _tx(z, snr_db, kind, K, block_len):
    """Transmit through fading+AWGN and return (received y, h, normalized symbols s)."""
    s = power_normalize(pack(z))
    B, m, _ = s.shape
    h = block_fading(B, m, block_len, z.device, kind, K)
    y = cmul(h, s)
    y, _ = awgn(y, snr_db)
    return y, h, s


def follower_jam(y, s, jsr_db, rho=0.3):
    """Add jamming power J/rho to the rho fraction of symbols with the most energy."""
    B, m, _ = s.shape
    J = 10.0 ** (jsr_db / 10.0)
    n = max(1, int(round(rho * m)))
    energy = (s ** 2).sum(-1)                       # (B,m)
    idx = energy.topk(n, dim=1).indices
    mask = torch.zeros(B, m, device=y.device).scatter_(1, idx, 1.0)
    per_sym = J / (n / m)                            # concentrate power
    jn = torch.randn_like(y) * (mask.unsqueeze(-1) * per_sym / 2.0).sqrt()
    return y + jn


def adversarial_jam(model, cond, y, h, x_target, jsr_db, steps=6, lr=0.3):
    """Worst-case power-constrained jam via projected gradient ascent on MSE."""
    B, m, _ = y.shape
    budget = math.sqrt(10.0 ** (jsr_db / 10.0) * m)  # L2 norm cap per sample
    delta = torch.zeros_like(y, requires_grad=True)
    for _ in range(steps):
        feats = torch.cat([y + delta, h], dim=-1)
        rec = model.dec(feats, cond)
        loss = ((rec - x_target) ** 2).mean()
        g, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta += lr * g / (g.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-8)
            norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1)
            factor = (budget / (norm + 1e-8)).clamp(max=1.0)
            delta *= factor
        delta.requires_grad_(True)
    return (y + delta).detach()
