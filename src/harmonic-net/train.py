"""
Train every model variant on the adding problem and log test MSE over training.

Runs the full experimental matrix:
    MLP          - no memory structure (control: should fail)
    GRU          - standard gated recurrent memory (strong baseline)
    HRM-full     - oscillatory memory + selective damping gate (the proposal)
    HRM-noGate   - ablation: novel selective gate removed
    HRM-noOsc    - ablation: resonance (rotation) removed

Writes results.json for the report generator.
"""

import json
import time
import torch

from data import make_batch, constant_baseline_mse
from model import HRM, GRUBaseline, MLPBaseline, count_params

# ---- config -------------------------------------------------------------
SEQ_LEN = 120
BATCH = 128
STEPS = 2000
EVAL_EVERY = 100
EVAL_BATCH = 2000
LR = 3e-3
SEED = 0
DEVICE = "cpu"  # tiny elementwise scan -> CPU avoids MPS dispatch overhead
# ------------------------------------------------------------------------


def build_models():
    return {
        "MLP": MLPBaseline(d_in=2, T=SEQ_LEN, hidden=128),
        "GRU": GRUBaseline(d_in=2, hidden=48),
        "HRM-full": HRM(d_in=2, n_channels=48, selective=True, oscillate=True),
        "HRM-noGate": HRM(d_in=2, n_channels=48, selective=False, oscillate=True),
        "HRM-noOsc": HRM(d_in=2, n_channels=48, selective=True, oscillate=False),
    }


def evaluate(model, gen):
    model.eval()
    with torch.no_grad():
        x, y = make_batch(EVAL_BATCH, SEQ_LEN, device=DEVICE, generator=gen)
        pred = model(x)
        mse = torch.mean((pred - y) ** 2).item()
    model.train()
    return mse


def train_one(name, model):
    torch.manual_seed(SEED)
    train_gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    eval_gen = torch.Generator(device=DEVICE).manual_seed(12345)  # fixed test set

    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    curve = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = make_batch(BATCH, SEQ_LEN, device=DEVICE, generator=train_gen)
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            mse = evaluate(model, eval_gen)
            curve.append({"step": step, "test_mse": mse})
            print(f"  [{name:11s}] step {step:5d}  test_mse {mse:.4f}")

    secs = time.time() - t0
    final = curve[-1]["test_mse"]
    return {
        "params": count_params(model),
        "final_mse": final,
        "curve": curve,
        "train_secs": round(secs, 1),
    }


def main():
    print(f"Adding problem  T={SEQ_LEN}  steps={STEPS}  device={DEVICE}")
    baseline = constant_baseline_mse()
    print(f"Constant-prediction baseline MSE = {baseline:.4f}\n")

    results = {
        "config": {
            "seq_len": SEQ_LEN,
            "batch": BATCH,
            "steps": STEPS,
            "lr": LR,
            "constant_baseline_mse": baseline,
        },
        "models": {},
    }

    for name, model in build_models().items():
        print(f"Training {name} ({count_params(model):,} params)")
        results["models"][name] = train_one(name, model)
        print()

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")

    print("\n=== SUMMARY (final test MSE, lower is better) ===")
    for name, r in sorted(results["models"].items(), key=lambda kv: kv[1]["final_mse"]):
        solved = "SOLVED" if r["final_mse"] < 0.02 else ""
        print(f"  {name:11s}  MSE {r['final_mse']:.4f}   {r['params']:>7,} params  {solved}")


if __name__ == "__main__":
    main()
