"""
Train the models on the frequency-identification task (classification).

Matrix:
    GRU          - standard gated recurrent memory (strong baseline)
    HRM-full     - resonant memory + selective damping gate (the proposal)
    HRM-noGate   - ablation: novel gate removed
    HRM-noOsc    - ablation: resonance removed (should collapse toward chance)

Writes results_freq.json.
"""

import json
import time
import torch
import torch.nn.functional as F

import data_freq as D
from model import HRM, GRUBaseline, count_params

SEQ_LEN = 100
BATCH = 128
STEPS = 1200
EVAL_EVERY = 100
EVAL_BATCH = 2000
LR = 3e-3
SEED = 0
DEVICE = "cpu"
NOISE = 2.0  # hard regime: enough noise to separate the architectures
K = D.N_CLASSES


def build_models():
    return {
        "GRU": GRUBaseline(d_in=1, hidden=48, out_dim=K),
        "HRM-full": HRM(d_in=1, n_channels=48, selective=True, oscillate=True, out_dim=K),
        "HRM-noGate": HRM(d_in=1, n_channels=48, selective=False, oscillate=True, out_dim=K),
        "HRM-noOsc": HRM(d_in=1, n_channels=48, selective=True, oscillate=False, out_dim=K),
    }


def evaluate(model, gen):
    model.eval()
    with torch.no_grad():
        x, y = D.make_batch(EVAL_BATCH, SEQ_LEN, noise=NOISE, device=DEVICE, generator=gen)
        logits = model(x)
        acc = (logits.argmax(-1) == y).float().mean().item()
    model.train()
    return acc


def train_one(name, model):
    torch.manual_seed(SEED)
    train_gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    eval_gen = torch.Generator(device=DEVICE).manual_seed(99999)  # fixed test set

    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    curve = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = D.make_batch(BATCH, SEQ_LEN, noise=NOISE, device=DEVICE, generator=train_gen)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            acc = evaluate(model, eval_gen)
            curve.append({"step": step, "test_acc": acc})
            print(f"  [{name:11s}] step {step:5d}  test_acc {acc:.3f}")

    return {
        "params": count_params(model),
        "final_acc": curve[-1]["test_acc"],
        "best_acc": max(p["test_acc"] for p in curve),
        "curve": curve,
        "train_secs": round(time.time() - t0, 1),
    }


def main():
    print(f"Frequency-ID  T={SEQ_LEN}  K={K}  noise={NOISE}  chance={D.chance_accuracy():.2f}\n")
    results = {
        "config": {"seq_len": SEQ_LEN, "n_classes": K, "noise": NOISE,
                   "steps": STEPS, "chance_acc": D.chance_accuracy()},
        "models": {},
    }
    for name, model in build_models().items():
        print(f"Training {name} ({count_params(model):,} params)")
        results["models"][name] = train_one(name, model)
        print()

    with open("results_freq.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote results_freq.json")

    print("\n=== SUMMARY (best test accuracy, higher is better) ===")
    for name, r in sorted(results["models"].items(), key=lambda kv: -kv[1]["best_acc"]):
        print(f"  {name:11s}  acc {r['best_acc']:.3f}   {r['params']:>7,} params")


if __name__ == "__main__":
    main()
