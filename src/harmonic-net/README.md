# HRM — Harmonic Resonance Mixer

A small, **from-scratch sequence-memory architecture** with a novel selective-resonance
mechanism — built, trained, and validated by controlled ablation. This is a real,
trainable model, not a wrapper around an existing one. No pretrained weights, no
external data downloads.

> **Honest scope.** I don't claim in an absolute sense that nothing like this has
> ever been built — the state-space / linear-RNN family (S4, S5, Mamba, LRU) is
> active research and shares DNA with the resonant half of this model. What this
> repo *does* is (1) implement a distinctive mechanism that isn't an off-the-shelf
> model, and (2) demonstrate empirically — with ablations and baselines — where it
> beats a standard RNN and where it doesn't, and *why*. The verifiable part is the
> evidence, and that's what's here.

## The mechanism

**1. Resonant memory (the "harmonic" part).** Each memory channel keeps a 2-D state
`s = (a, b)` that is *rotated* by a learnable angular frequency `ω` and shrunk by a
decay each timestep:

```
a_t = decay_t · ( cos(ω)·a_{t-1} − sin(ω)·b_{t-1} ) + drive_t
b_t = decay_t · ( sin(ω)·a_{t-1} + cos(ω)·b_{t-1} )
```

The rotation makes each channel a resonator tuned to a frequency; the bank is
initialised on a log-spaced spectrum so it tiles many timescales. Readout is
per-channel **energy** `√(a²+b²)` — the natural, phase-invariant signature of "did
this resonator light up".

**2. Selective damping gate (the novel component being tested).** Instead of a fixed
decay, decay is modulated *per channel, per timestep* by a content-based gate:

```
g_t     = sigmoid(W_g · x_t + b_g)              # ∈ (0,1), one per channel
decay_t = base_decay ^ g_t   where base_decay = exp(−softplus(λ))
```

`g_t → 0` ⇒ decay → 1: the channel **latches** and holds indefinitely. `g_t → 1` ⇒
it forgets at its base rate. So the network learns *when to grab and hold* specific
frequency channels based on what it's reading.

## Results (measured, this repo)

Two tasks, chosen to show both sides of the architecture's inductive bias.

### ✅ Where it wins — noisy frequency identification

Classify which of 5 frequencies is hidden in a noisy, random-phase signal (T=100).
Both models trained identically; test accuracy on a fixed 2000-example set.

| noise σ | HRM-full | GRU  |
|--------:|:--------:|:----:|
| 1.5     |  ~0.98   | ~0.90|
| 2.5     |  **~0.81** | ~0.49|

As noise rises the GRU collapses toward chance (0.20) while HRM stays strong — its
tuned-resonator prior integrates energy at the target frequency and rejects
broadband noise. **Ablation:** freezing the rotation (`HRM-noOsc`) drops accuracy to
~0.53, proving the resonance is what's doing the work, not the parameter budget.

### ⚠️ Where it loses — the adding problem (honest limitation)

Pure delayed-scalar recall (sum two marked values across T=120). Here the **GRU
solves it (MSE ≈ 0.0004) while every HRM variant stalls near the constant-guess
baseline (≈ 0.15–0.17)**. The reason is structural: an oscillator rotates a stored
value by `ω·(T−t)`, and because the marker→readout gap varies per example, the value
lands at a delay-dependent phase a linear head can't recover — and the resonator bank
can't select *which* inputs to write. Resonant memory is a **specialized prior** —
great for periodic/frequency structure, poor for arbitrary-delay recall. Showing both
tasks is the point.

## Run it

```bash
python3 train_freq.py    # frequency matrix + learning curves -> results_freq.json
python3 sweep.py         # noise sweep HRM vs GRU            -> results_sweep.json
python3 train.py         # adding problem (limitation)       -> results.json
python3 make_report.py   # renders report.html
open report.html
```

## Files

- `model.py` — HRM, the `SelectiveResonanceLayer` (oscillator bank + gate + energy readout), and MLP/GRU baselines.
- `data_freq.py` / `data.py` — the frequency-ID and adding-problem generators.
- `train_freq.py` / `sweep.py` / `train.py` — the three experiments.
- `make_report.py` — self-contained HTML report (hand-rolled SVG, no plotting deps).

## Limitations & honest notes

- Sequential Python scan (fine at these lengths; a parallel associative scan would
  be needed to scale to long sequences / GPUs efficiently).
- Small models, single seed, two synthetic tasks. A real research claim would need
  multiple seeds with error bars, more tasks, and comparison against modern SSMs
  (S4/S5/Mamba/LRU), which share the resonant half of this design.
- The novelty claim is deliberately narrow: a specific selective-damping-over-a-
  resonator-bank mechanism, shown by ablation to help on frequency-structured tasks.
