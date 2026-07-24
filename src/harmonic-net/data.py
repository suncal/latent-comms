"""
Synthetic long-range benchmark: the "Adding Problem" (Hochreiter & Schmidhuber, 1997).

This is the canonical stress test for long-range credit assignment in sequence
models. Each example is a length-T sequence with two channels:

    channel 0 (value)  ~ Uniform(0, 1) at every step
    channel 1 (marker) = 1.0 at exactly two random positions, 0.0 elsewhere

The target is the SUM of the two values whose marker == 1. To solve it, a model
must (a) detect *when* a marker fires and (b) carry the associated value across
potentially the whole sequence to the end. Models with no memory, or with memory
that decays too fast, cannot do this.

Baseline reference: predicting the constant mean (1.0) gives MSE == Var(v1+v2)
== 2 * (1/12) ~= 0.1667. Any model well below that has learned something; a model
that "solves" the task lands near MSE < 0.02.
"""

import torch


def make_batch(batch_size, T, device="cpu", generator=None):
    """Return (x, y) where x is (B, T, 2) and y is (B, 1)."""
    values = torch.rand(batch_size, T, 1, generator=generator, device=device)
    markers = torch.zeros(batch_size, T, 1, device=device)

    # Two distinct marked positions per sequence. First in the early half,
    # second anywhere after it, so the gap is frequently long.
    first = torch.randint(0, T // 2, (batch_size,), generator=generator, device=device)
    second = torch.randint(0, T, (batch_size,), generator=generator, device=device)
    # Ensure the two positions differ.
    clash = second == first
    second[clash] = (second[clash] + 1) % T

    rows = torch.arange(batch_size, device=device)
    markers[rows, first, 0] = 1.0
    markers[rows, second, 0] = 1.0

    x = torch.cat([values, markers], dim=-1)  # (B, T, 2)
    y = (values[rows, first, 0] + values[rows, second, 0]).unsqueeze(-1)  # (B, 1)
    return x, y


# MSE achieved by always predicting the mean of the target distribution.
def constant_baseline_mse():
    return 2.0 * (1.0 / 12.0)
