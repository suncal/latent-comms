"""
HRM - Harmonic Resonance Mixer
==============================

A small sequence model whose memory is a bank of learnable *damped oscillators*.
Each memory channel maintains a 2-D state s = (a, b) that, every timestep, is
rotated by a learnable angular frequency omega and shrunk by a decay factor:

    a_t = decay_t * ( cos(omega)*a_{t-1} - sin(omega)*b_{t-1} ) + drive_t
    b_t = decay_t * ( sin(omega)*a_{t-1} + cos(omega)*b_{t-1} )

That rotation makes each channel a resonator tuned to a particular frequency
(this is the "resonance" half of the name). Standard diagonal linear-RNN / SSM
memories use a real decay only; the rotation lets a channel *hold and phase-track*
periodic or delayed structure instead of just leaking exponentially.

The NOVEL component being tested is the **selective damping gate**. Instead of a
fixed decay per channel, the decay at each step is modulated by a content-based
gate computed from the current input:

    g_t     = sigmoid( W_g x_t + b_g )                # in (0, 1), per channel
    decay_t = exp( -softplus(log_lambda) * g_t )      # per channel, per step

When g_t -> 0 the decay -> 1: the channel *latches*, holding its current state
indefinitely regardless of its base leak rate. When g_t -> 1 the channel decays
at its full base rate exp(-softplus(log_lambda)). So the network learns to open
the gate ("forget / overwrite") or close it ("remember this") on a per-channel,
per-timestep basis driven by what it is currently reading. On the adding problem
the useful behaviour is: latch the value channels when a marker fires, hold them
to the end.

Ablations (same code path, flags flip one mechanism off) let us isolate each idea:
  * selective=False  -> gate frozen open (g_t = 1): plain per-channel decay.
                        Removes the NOVEL component.
  * oscillate=False  -> omega frozen at 0: no rotation, pure leaky memory.
                        Removes the resonance.

The scan is written with pure elementwise ops over the (B, N) state (no matmul
inside the time loop), so it runs fast on CPU for the sequence lengths here.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveResonanceLayer(nn.Module):
    def __init__(self, d_in, n_channels, selective=True, oscillate=True):
        super().__init__()
        self.n = n_channels
        self.selective = selective
        self.oscillate = oscillate

        # Input -> per-channel real drive added to each oscillator.
        self.drive = nn.Linear(d_in, n_channels)
        # Input -> per-channel gate logits (only used when selective=True).
        self.gate = nn.Linear(d_in, n_channels)

        # Learnable base leak rate per channel (via softplus -> positive).
        # Init spread so channels range from fast-leaking to near-lossless.
        self.log_lambda = nn.Parameter(torch.linspace(-3.0, 0.5, n_channels))

        # Learnable resonant frequency per channel. Init on a log-spaced spectrum
        # from very slow to ~half-Nyquist so the bank tiles a range of timescales.
        omega0 = torch.logspace(math.log10(0.005), math.log10(1.5), n_channels)
        self.omega = nn.Parameter(omega0)

        # Bias the gate closed at init (prefer holding), so early training keeps
        # information around long enough to get useful gradients.
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, x):
        # x: (B, T, d_in)
        B, T, _ = x.shape
        device = x.device

        if self.oscillate:
            cos = torch.cos(self.omega)
            sin = torch.sin(self.omega)
        else:
            cos = torch.ones(self.n, device=device)
            sin = torch.zeros(self.n, device=device)

        base_decay = torch.exp(-F.softplus(self.log_lambda))  # (N,)

        drive = self.drive(x)  # (B, T, N)
        if self.selective:
            g = torch.sigmoid(self.gate(x))  # (B, T, N) in (0,1)
        else:
            g = None

        a = torch.zeros(B, self.n, device=device)
        b = torch.zeros(B, self.n, device=device)

        for t in range(T):
            if self.selective:
                # decay_t = base_decay ** g_t  (elementwise); g_t in (0,1)
                # g->0 => decay->1 (latch); g->1 => decay->base_decay.
                decay = base_decay.pow(g[:, t, :])  # (B, N)
            else:
                decay = base_decay.unsqueeze(0)  # (1, N)

            ra = cos * a - sin * b
            rb = sin * a + cos * b
            a = decay * ra + drive[:, t, :]
            b = decay * rb

        # Readout: raw final state PLUS per-channel energy sqrt(a^2+b^2).
        # Energy is the natural, phase-invariant readout of a resonator: a channel
        # tuned to a frequency present in the input builds large energy regardless
        # of phase, so downstream layers can read "which resonators lit up".
        energy = torch.sqrt(a * a + b * b + 1e-6)
        return torch.cat([a, b, energy], dim=-1)  # (B, 3N)


class HRM(nn.Module):
    """Harmonic Resonance Mixer for sequence -> scalar regression."""

    def __init__(self, d_in=2, n_channels=48, selective=True, oscillate=True, out_dim=1):
        super().__init__()
        self.mem = SelectiveResonanceLayer(d_in, n_channels, selective, oscillate)
        self.head = nn.Sequential(
            nn.Linear(3 * n_channels, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.head(self.mem(x))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class GRUBaseline(nn.Module):
    def __init__(self, d_in=2, hidden=48, out_dim=1):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class MLPBaseline(nn.Module):
    """No temporal structure: flatten the whole sequence and regress."""

    def __init__(self, d_in=2, T=120, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_in * T, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
