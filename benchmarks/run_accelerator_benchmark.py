"""
AcceleratorAI Comprehensive Empirical Benchmark Suite.

Evaluates the 4 core pillars of the VIBE Turbine Learning Engine:
  1. Convergence Acceleration Factor (AF = Steps_baseline / Steps_accelerator)
  2. Data Efficiency (Validation Accuracy vs Training Dataset Size)
  3. Local Minima Trap Escape Rate (% of runs escaping deceptive saddle basins)
  4. Representation Robustness & Generalization Under Noise (OOD Noise Resistance)

Compares 3 Architectures:
  [A] Baseline: Standard mini-batch SGD with momentum
  [B] Turbo Only: Mechanical Drive Shaft + Centrifugal Compressor (No Injectors)
  [C] Full AcceleratorAI: Mechanical Shaft + Compressor + Asynchronous Injectors + DNA Plecionka
"""

import os
import sys
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
import numpy as np

from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.engine import TurboLearningEngine


# ---------------------------------------------------------------------------
# Synthetic Benchmark Manifolds
# ---------------------------------------------------------------------------

def generate_spiral_dataset(n_samples: int = 1200, noise: float = 0.12, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates an interlocking double spiral manifold (classic non-linear benchmark)."""
    np.random.seed(seed)
    n = n_samples // 2

    # Spiral 1
    theta1 = np.sqrt(np.random.rand(n)) * 3.0 * np.pi
    r1 = 2.0 * theta1 + np.pi
    x1 = np.stack([np.cos(theta1) * r1, np.sin(theta1) * r1], axis=1) + np.random.randn(n, 2) * noise * 10
    y1 = np.zeros(n, dtype=np.int32)

    # Spiral 2
    theta2 = np.sqrt(np.random.rand(n)) * 3.0 * np.pi
    r2 = -2.0 * theta2 - np.pi
    x2 = np.stack([np.cos(theta2) * r2, np.sin(theta2) * r2], axis=1) + np.random.randn(n, 2) * noise * 10
    y2 = np.ones(n, dtype=np.int32)

    x = np.concatenate([x1, x2], axis=0).astype(np.float32)
    y = np.concatenate([y1, y2], axis=0).astype(np.int32)

    # Normalize to zero mean, unit variance
    x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-6)

    perm = np.random.permutation(n_samples)
    return x[perm], y[perm]


def generate_deceptive_trap_dataset(n_samples: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a multimodal dataset with a deceptive local minimum basin.
    Standard optimizers get easily trapped predicting the dominant majority false mode.
    """
    np.random.seed(seed)
    # Mode 0 (True class 0): 40%
    n0 = int(n_samples * 0.45)
    x0 = np.random.randn(n0, 4) * 0.7 + np.array([-1.8, -1.8, 0.5, -0.5])
    y0 = np.zeros(n0, dtype=np.int32)

    # Mode 1 (True class 1 - complex ring): 45%
    n1 = int(n_samples * 0.45)
    angle = np.random.uniform(0, 2 * np.pi, n1)
    r = 2.2 + np.random.normal(0, 0.2, n1)
    x1 = np.zeros((n1, 4))
    x1[:, 0] = np.cos(angle) * r
    x1[:, 1] = np.sin(angle) * r
    x1[:, 2] = np.random.randn(n1) * 0.5
    x1[:, 3] = np.random.randn(n1) * 0.5
    y1 = np.ones(n1, dtype=np.int32)

    # Deceptive Saddle Trap (10% deceptive decoy cluster)
    n_trap = n_samples - n0 - n1
    x_trap = np.random.randn(n_trap, 4) * 0.2 + np.array([0.0, 0.0, 0.0, 0.0])
    y_trap = np.zeros(n_trap, dtype=np.int32)

    x = np.concatenate([x0, x1, x_trap], axis=0).astype(np.float32)
    y = np.concatenate([y0, y1, y_trap], axis=0).astype(np.int32)

    perm = np.random.permutation(n_samples)
    return x[perm], y[perm]


# ---------------------------------------------------------------------------
# Training Harnesses
# ---------------------------------------------------------------------------

def train_baseline(x_train, y_train, epochs=25, batch_size=32, lr=0.015, loss_target=None):
    """Standard mini-batch SGD with momentum."""
    model = PureNumPyMLP(layer_sizes=[x_train.shape[1], 32, 16, 2], activation="tanh", momentum=0.9)
    n = len(x_train)
    losses = []
    steps_to_target = None
    step_count = 0

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            step_count += 1
            idx = perm[i : i + batch_size]
            _, loss = model.forward_and_loss(x_train[idx], y_train[idx])
            model.backward()
            model.apply_updates(learning_rate=lr)
            epoch_loss += loss
            batches += 1

            if loss_target is not None and loss <= loss_target and steps_to_target is None:
                steps_to_target = step_count

        losses.append(epoch_loss / max(1, batches))

    return {
        "model": model,
        "final_loss": losses[-1],
        "losses": losses,
        "steps_to_target": steps_to_target or step_count,
        "reached_target": steps_to_target is not None,
    }


def train_turbo_only(x_train, y_train, epochs=25, batch_size=32, base_lr=0.015, loss_target=None):
    """Mechanical Drive Shaft + Centrifugal Compressor (No async injectors)."""
    model = PureNumPyMLP(layer_sizes=[x_train.shape[1], 32, 16, 2], activation="tanh", momentum=0.9)
    engine = TurboLearningEngine(
        model=model,
        base_learning_rate=base_lr,
        target_boost_psi=14.7,
        enable_default_injectors=False,
    )
    n = len(x_train)
    losses = []
    steps_to_target = None
    step_count = 0

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            step_count += 1
            idx = perm[i : i + batch_size]
            res = engine.step(x_train[idx], y_train[idx])
            epoch_loss += res.loss
            batches += 1

            if loss_target is not None and res.loss <= loss_target and steps_to_target is None:
                steps_to_target = step_count

        losses.append(epoch_loss / max(1, batches))

    return {
        "model": model,
        "engine": engine,
        "final_loss": losses[-1],
        "losses": losses,
        "steps_to_target": steps_to_target or step_count,
        "reached_target": steps_to_target is not None,
        "final_rpm": engine.shaft.rpm,
    }


def train_full_accelerator(x_train, y_train, epochs=25, batch_size=32, base_lr=0.015, loss_target=None):
    """Full AcceleratorAI: Shaft + Compressor + Async Injectors + DNA Plecionka."""
    model = PureNumPyMLP(layer_sizes=[x_train.shape[1], 32, 16, 2], activation="tanh", momentum=0.9)
    engine = TurboLearningEngine(
        model=model,
        base_learning_rate=base_lr,
        target_boost_psi=14.7,
        enable_default_injectors=True,
    )
    # Feed reservoir with training data
    for inj in engine.injectors:
        if hasattr(inj, "set_reservoir"):
            inj.set_reservoir(x_train[:80], y_train[:80])

    n = len(x_train)
    losses = []
    steps_to_target = None
    step_count = 0

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            step_count += 1
            idx = perm[i : i + batch_size]
            res = engine.step(x_train[idx], y_train[idx])
            epoch_loss += res.loss
            batches += 1

            if loss_target is not None and res.loss <= loss_target and steps_to_target is None:
                steps_to_target = step_count

        losses.append(epoch_loss / max(1, batches))

    return {
        "model": model,
        "engine": engine,
        "final_loss": losses[-1],
        "losses": losses,
        "steps_to_target": steps_to_target or step_count,
        "reached_target": steps_to_target is not None,
        "final_rpm": engine.shaft.rpm,
        "helical_resonance": engine.braided_ecu.resonance_index,
    }


def evaluate_accuracy(model, x_test, y_test):
    preds, _ = model.forward_and_loss(x_test, y_test)
    acc = np.mean(np.argmax(preds, axis=-1) == y_test) * 100.0
    return float(acc)


# ---------------------------------------------------------------------------
# Benchmark 1: Acceleration Factor & Convergence Speed
# ---------------------------------------------------------------------------

def run_acceleration_benchmark(n_trials: int = 5) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print(" [BENCHMARK 1] CONVERGENCE ACCELERATION FACTOR (AF)")
    print(" Testing epochs required to reach 65.0% Validation Accuracy on double spiral")
    print("=" * 76)

    target_acc = 65.0
    results = {"baseline": [], "turbo_only": [], "full_accelerator": []}
    final_accs = {"baseline": [], "turbo_only": [], "full_accelerator": []}

    for trial in range(n_trials):
        seed = 100 + trial
        x_data, y_data = generate_spiral_dataset(n_samples=1200, seed=seed)
        split = 900
        x_train, y_train = x_data[:split], y_data[:split]
        x_val, y_val = x_data[split:], y_data[split:]

        # Baseline
        base_epochs = None
        base_model = PureNumPyMLP(layer_sizes=[2, 32, 16, 2], activation="tanh", momentum=0.9)
        for ep in range(1, 26):
            perm = np.random.permutation(len(x_train))
            for i in range(0, len(x_train), 32):
                idx = perm[i : i + 32]
                base_model.forward_and_loss(x_train[idx], y_train[idx])
                base_model.backward()
                base_model.apply_updates(0.015)
            acc = evaluate_accuracy(base_model, x_val, y_val)
            if acc >= target_acc and base_epochs is None:
                base_epochs = ep
        final_accs["baseline"].append(acc)
        results["baseline"].append(base_epochs or 25)

        # Turbo Only
        turbo_epochs = None
        turbo_model = PureNumPyMLP(layer_sizes=[2, 32, 16, 2], activation="tanh", momentum=0.9)
        turbo_engine = TurboLearningEngine(turbo_model, base_learning_rate=0.015, enable_default_injectors=False)
        for ep in range(1, 26):
            turbo_engine.train_epoch(x_train, y_train, batch_size=32, shuffle=True)
            acc = evaluate_accuracy(turbo_model, x_val, y_val)
            if acc >= target_acc and turbo_epochs is None:
                turbo_epochs = ep
        final_accs["turbo_only"].append(acc)
        results["turbo_only"].append(turbo_epochs or 25)

        # Full AcceleratorAI
        full_epochs = None
        full_model = PureNumPyMLP(layer_sizes=[2, 32, 16, 2], activation="tanh", momentum=0.9)
        full_engine = TurboLearningEngine(full_model, base_learning_rate=0.015, enable_default_injectors=True)
        for ep in range(1, 26):
            full_engine.train_epoch(x_train, y_train, batch_size=32, shuffle=True)
            acc = evaluate_accuracy(full_model, x_val, y_val)
            if acc >= target_acc and full_epochs is None:
                full_epochs = ep
        final_accs["full_accelerator"].append(acc)
        results["full_accelerator"].append(full_epochs or 25)

        print(f"  Trial #{trial+1:02d} | Baseline: {results['baseline'][-1]:2d} ep ({final_accs['baseline'][-1]:.1f}%) | "
              f"Turbo: {results['turbo_only'][-1]:2d} ep ({final_accs['turbo_only'][-1]:.1f}%) | "
              f"Full AcceleratorAI: {results['full_accelerator'][-1]:2d} ep ({final_accs['full_accelerator'][-1]:.1f}%)")

    mean_base = float(np.mean(results["baseline"]))
    mean_turbo = float(np.mean(results["turbo_only"]))
    mean_full = float(np.mean(results["full_accelerator"]))

    mean_acc_base = float(np.mean(final_accs["baseline"]))
    mean_acc_turbo = float(np.mean(final_accs["turbo_only"]))
    mean_acc_full = float(np.mean(final_accs["full_accelerator"]))

    af_full = mean_base / max(1.0, mean_full)

    print("-" * 76)
    print(f"  Avg Epochs to {target_acc}% Val Acc: Baseline={mean_base:.1f} | Turbo Only={mean_turbo:.1f} | Full AcceleratorAI={mean_full:.1f}")
    print(f"  Avg Final Validation Accuracy: Baseline={mean_acc_base:.1f}% | Turbo Only={mean_acc_turbo:.1f}% | Full AcceleratorAI={mean_acc_full:.1f}%")
    print(f"  >>> ACCELERATION FACTOR (AF): {af_full:.2f}x Speedup | Accuracy Advantage: +{mean_acc_full - mean_acc_base:.1f}%")
    return {
        "mean_epochs_baseline": mean_base,
        "mean_epochs_turbo": mean_turbo,
        "mean_epochs_full": mean_full,
        "mean_acc_baseline": mean_acc_base,
        "mean_acc_turbo": mean_acc_turbo,
        "mean_acc_full": mean_acc_full,
        "af_full": round(af_full, 2),
    }


# ---------------------------------------------------------------------------
# Benchmark 2: Data Efficiency Test
# ---------------------------------------------------------------------------

def run_data_efficiency_benchmark() -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print(" [BENCHMARK 2] DATA EFFICIENCY TEST")
    print(" Measuring test accuracy when training dataset is restricted (25%, 50%, 75%, 100%)")
    print("=" * 76)

    fractions = [0.25, 0.50, 0.75, 1.00]
    x_all, y_all = generate_spiral_dataset(n_samples=1200, seed=42)
    x_test, y_test = x_all[900:], y_all[900:]
    x_train_full, y_train_full = x_all[:900], y_all[:900]

    out_base = []
    out_full = []

    print(f"| Data Fraction | Training Samples | Baseline Accuracy | Full AcceleratorAI | Difference |")
    print(f"|---------------|------------------|-------------------|--------------------|------------|")

    for frac in fractions:
        n_sub = int(len(x_train_full) * frac)
        x_sub, y_sub = x_train_full[:n_sub], y_train_full[:n_sub]

        base_res = train_baseline(x_sub, y_sub, epochs=20)
        full_res = train_full_accelerator(x_sub, y_sub, epochs=20)

        acc_base = evaluate_accuracy(base_res["model"], x_test, y_test)
        acc_full = evaluate_accuracy(full_res["model"], x_test, y_test)

        diff = acc_full - acc_base
        sign = "+" if diff >= 0 else ""
        out_base.append(acc_base)
        out_full.append(acc_full)

        print(f"|    {int(frac*100):3d}%      |      {n_sub:4d}        |      {acc_base:5.1f}%       |       {acc_full:5.1f}%       |   {sign}{diff:4.1f}%   |")

    print("-" * 76)
    # Check if AcceleratorAI with 50% data matches or beats Baseline with 100% data
    print(f"  AcceleratorAI @ 50% data: {out_full[1]:.1f}%  vs  Baseline @ 100% data: {out_base[3]:.1f}%")
    return {
        "fractions": fractions,
        "baseline_accuracies": out_base,
        "full_accuracies": out_full,
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Local Minima & Saddle-Point Trap Escape
# ---------------------------------------------------------------------------

def run_trap_escape_benchmark(n_trials: int = 10) -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print(" [BENCHMARK 3] LOCAL MINIMA & DECEPTIVE TRAP ESCAPE RATE")
    print(" Testing ability of Asynchronous Chaos Shock to bust out of multimodal traps")
    print("=" * 76)

    escapes_base = 0
    escapes_turbo = 0
    escapes_full = 0
    threshold_loss = 0.38  # Escape threshold: successfully finding non-trivial mode

    for trial in range(n_trials):
        seed = 200 + trial
        x_data, y_data = generate_deceptive_trap_dataset(n_samples=1000, seed=seed)
        split = 800
        x_train, y_train = x_data[:split], y_data[:split]

        res_b = train_baseline(x_train, y_train, epochs=25)
        res_t = train_turbo_only(x_train, y_train, epochs=25)
        res_f = train_full_accelerator(x_train, y_train, epochs=25)

        esc_b = res_b["final_loss"] < threshold_loss
        esc_t = res_t["final_loss"] < threshold_loss
        esc_f = res_f["final_loss"] < threshold_loss

        if esc_b: escapes_base += 1
        if esc_t: escapes_turbo += 1
        if esc_f: escapes_full += 1

        b_tag = "ESCAPED" if esc_b else "STUCK  "
        t_tag = "ESCAPED" if esc_t else "STUCK  "
        f_tag = "ESCAPED" if esc_f else "STUCK  "

        print(f"  Trial #{trial+1:02d} | Baseline: {b_tag} ({res_b['final_loss']:.3f}) | "
              f"Turbo: {t_tag} ({res_t['final_loss']:.3f}) | "
              f"Full AcceleratorAI: {f_tag} ({res_f['final_loss']:.3f})")

    rate_base = (escapes_base / n_trials) * 100.0
    rate_turbo = (escapes_turbo / n_trials) * 100.0
    rate_full = (escapes_full / n_trials) * 100.0

    print("-" * 76)
    print(f"  TRAP ESCAPE SUCCESS RATE:")
    print(f"  - Baseline (Standard SGD):    {rate_base:5.1f}%  ({escapes_base}/{n_trials})")
    print(f"  - Turbo Only:                 {rate_turbo:5.1f}%  ({escapes_turbo}/{n_trials})")
    print(f"  - Full AcceleratorAI (Chaos): {rate_full:5.1f}%  ({escapes_full}/{n_trials})")
    return {
        "escape_rate_baseline": rate_base,
        "escape_rate_turbo": rate_turbo,
        "escape_rate_full": rate_full,
    }


# ---------------------------------------------------------------------------
# Benchmark 4: Noise & OOD Robustness
# ---------------------------------------------------------------------------

def run_noise_robustness_benchmark() -> Dict[str, Any]:
    print("\n" + "=" * 76)
    print(" [BENCHMARK 4] NOISE & OUT-OF-DISTRIBUTION ROBUSTNESS")
    print(" Corrupting test data with Gaussian perturbation noise (sigma: 0.0 -> 0.40)")
    print("=" * 76)

    x_all, y_all = generate_spiral_dataset(n_samples=1200, seed=42)
    split = 900
    x_train, y_train = x_all[:split], y_all[:split]
    x_test, y_test = x_all[split:], y_all[split:]

    base_res = train_baseline(x_train, y_train, epochs=25)
    full_res = train_full_accelerator(x_train, y_train, epochs=25)

    noise_levels = [0.0, 0.10, 0.20, 0.35]
    res_base_accs = []
    res_full_accs = []

    print(f"| Noise Level (Sigma) | Baseline Accuracy | Full AcceleratorAI | Advantage |")
    print(f"|---------------------|-------------------|--------------------|-----------|")

    for sigma in noise_levels:
        noise = np.random.randn(*x_test.shape) * sigma
        x_corrupted = x_test + noise

        acc_b = evaluate_accuracy(base_res["model"], x_corrupted, y_test)
        acc_f = evaluate_accuracy(full_res["model"], x_corrupted, y_test)
        adv = acc_f - acc_b

        res_base_accs.append(acc_b)
        res_full_accs.append(acc_f)

        print(f"|      sigma={sigma:.2f}     |      {acc_b:5.1f}%       |       {acc_f:5.1f}%       |  +{adv:4.1f}%   |")

    print("-" * 76)
    mean_adv = float(np.mean(np.array(res_full_accs) - np.array(res_base_accs)))
    print(f"  Average Accuracy Advantage under noise: +{mean_adv:.2f}%")
    return {
        "noise_levels": noise_levels,
        "baseline_accuracy": res_base_accs,
        "full_accuracy": res_full_accs,
        "mean_advantage": round(mean_adv, 2),
    }


# ---------------------------------------------------------------------------
# Main Execution Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("   ACCELERATOR-AI — RIGOROUS EMPIRICAL BENCHMARK SUITE")
    print("   Testing Physical Feedback Loop 1.0 & DNA Braided Dynamics")
    print("=" * 76)
    t_start = time.time()

    b1 = run_acceleration_benchmark(n_trials=5)
    b2 = run_data_efficiency_benchmark()
    b3 = run_trap_escape_benchmark(n_trials=10)
    b4 = run_noise_robustness_benchmark()

    total_time = time.time() - t_start

    print("\n" + "=" * 76)
    print("   FINAL BENCHMARK SYNTHESIS: IS IT TRULY AN AI ACCELERATOR?")
    print("=" * 76)
    print(f"  1. Speed Acceleration Factor:       {b1['af_full']:.2f}x faster convergence to target")
    print(f"  2. Local Minima Escape Rate:        {b3['escape_rate_full']:.1f}% vs {b3['escape_rate_baseline']:.1f}% (Baseline)")
    print(f"  3. Noise Resistance Advantage:      +{b4['mean_advantage']:.1f}% higher generalization accuracy")
    print(f"  Total Benchmark Execution Time:     {total_time:.2f} seconds")
    print("=" * 76)

    # Save results to JSON artifact
    os.makedirs("benchmarks/results", exist_ok=True)
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "acceleration_factor": b1,
        "data_efficiency": b2,
        "trap_escape": b3,
        "noise_robustness": b4,
        "total_duration_s": round(total_time, 2),
    }

    with open("benchmarks/results/latest_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(">>> Benchmark artifact exported to benchmarks/results/latest_benchmark.json\n")


if __name__ == "__main__":
    main()
