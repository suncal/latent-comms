"""
Frequency-identification benchmark (the task a resonant memory is *built* for).

Each example is a length-T 1-D signal: a single sinusoid at one of K frequencies,
with a RANDOM PHASE and additive Gaussian noise. The label is which frequency
class produced it. To win, a model must extract periodic structure across the
whole sequence and be invariant to phase and robust to noise.

    x_t = sin(2*pi * f_k * t / T + phi) + noise,   phi ~ U(0, 2pi)

This is exactly the inductive bias a bank of tuned oscillators encodes: the
channel whose frequency matches f_k resonates (accumulates energy) while the
others stay quiet. A standard RNN has to *learn* to count oscillations from
scratch, which is much harder under noise and random phase.

Chance accuracy = 1/K. Input dim is 1 (a scalar per timestep).
"""

import math
import torch

N_CLASSES = 5
# Frequencies (cycles over the whole window) the classes correspond to.
FREQS = torch.tensor([3.0, 5.0, 8.0, 12.0, 17.0])


def make_batch(batch_size, T, noise=0.5, device="cpu", generator=None):
    labels = torch.randint(0, N_CLASSES, (batch_size,), generator=generator, device=device)
    f = FREQS.to(device)[labels]  # (B,)
    t = torch.arange(T, device=device).float() / T  # (T,)
    phase = torch.rand(batch_size, 1, generator=generator, device=device) * 2 * math.pi
    signal = torch.sin(2 * math.pi * f.unsqueeze(1) * t.unsqueeze(0) + phase)  # (B, T)
    signal = signal + noise * torch.randn(batch_size, T, generator=generator, device=device)
    x = signal.unsqueeze(-1)  # (B, T, 1)
    return x, labels


def chance_accuracy():
    return 1.0 / N_CLASSES
