"""
Demo Benchmark: Standard Linear Training vs. Turbocharged AcceleratorAI

Demonstrates how the Turbocharged Engine with Asynchronous Multi-Point Injectors
breaks out of local minima and reaches convergence faster with fewer steps.
"""

import time
import numpy as np
from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.engine import TurboLearningEngine


def generate_benchmark_dataset(n_samples: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a non-linear 2D classification problem with deceptive local minima
    (two concentric spirals / multi-modal manifold).
    """
    np.random.seed(42)
    n_per_class = n_samples // 2

    # Class 0: Inner swirling ring
    r0 = np.linspace(0.2, 1.2, n_per_class)
    theta0 = np.linspace(0, 2.5 * np.pi, n_per_class) + np.random.normal(0, 0.1, n_per_class)
    x0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)
    y0 = np.zeros(n_per_class)

    # Class 1: Outer swirling ring
    r1 = np.linspace(0.6, 1.8, n_per_class)
    theta1 = np.linspace(np.pi, 3.5 * np.pi, n_per_class) + np.random.normal(0, 0.1, n_per_class)
    x1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
    y1 = np.ones(n_per_class)

    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([y0, y1], axis=0)

    # Shuffle
    perm = np.random.permutation(n_samples)
    return x[perm].astype(np.float32), y[perm].astype(np.int32)


def run_baseline_training(x_train, y_train, epochs=25, batch_size=32, lr=0.01):
    """Standard conventional training loop (no turbo boost, no async injectors)."""
    np.random.seed(42)
    model = PureNumPyMLP(layer_sizes=[2, 32, 16, 2], activation="tanh", momentum=0.9)
    n = len(x_train)
    losses = []

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            xb = x_train[i : i + batch_size]
            yb = y_train[i : i + batch_size]
            _, loss = model.forward_and_loss(xb, yb)
            model.backward()
            model.apply_updates(learning_rate=lr)
            epoch_loss += loss
            batches += 1
        losses.append(epoch_loss / batches)

    duration = time.time() - start_time
    return model, losses, duration


def run_turbocharged_training(x_train, y_train, epochs=25, batch_size=32, base_lr=0.01):
    """AcceleratorAI Turbocharged training loop with active async injectors & ECU."""
    np.random.seed(42)
    model = PureNumPyMLP(layer_sizes=[2, 32, 16, 2], activation="tanh", momentum=0.9)
    engine = TurboLearningEngine(
        model=model,
        target_boost_psi=14.7,  # 1.0 bar boost
        base_learning_rate=base_lr,
        enable_default_injectors=True,
    )

    # Set real-world reservoir
    for inj in engine.injectors:
        if hasattr(inj, "set_reservoir"):
            inj.set_reservoir(x_train[:100], y_train[:100])

    losses = []
    telemetry_history = []

    engine.telemetry_hub.add_listener(lambda t: telemetry_history.append(t))

    start_time = time.time()
    for epoch in range(epochs):
        epoch_losses = engine.train_epoch(x_train, y_train, batch_size=batch_size, shuffle=True)
        losses.append(float(np.mean(epoch_losses)))

    duration = time.time() - start_time
    return engine, losses, duration, telemetry_history


def main():
    print("=" * 70)
    print("   ACCELERATOR-AI — TURBOCHARGED ENGINE VS. STANDARD BENCHMARK")
    print("=" * 70)

    x_data, y_data = generate_benchmark_dataset(n_samples=1000)
    train_split = 800
    x_train, y_train = x_data[:train_split], y_data[:train_split]
    x_val, y_val = x_data[train_split:], y_data[train_split:]

    print(f"[*] Dataset: 2-Spiral Non-Convex Manifold ({len(x_train)} train, {len(x_val)} val)")
    print("[*] Running Baseline Training (No Turbo, No Injectors)...")
    base_model, base_losses, base_time = run_baseline_training(x_train, y_train, epochs=20)

    print("[*] Running AcceleratorAI Turbo Engine (Compressor + Async Injectors + ECU)...")
    turbo_engine, turbo_losses, turbo_time, telemetry = run_turbocharged_training(x_train, y_train, epochs=20)

    # Evaluate validation accuracy
    base_preds, base_val_loss = base_model.forward_and_loss(x_val, y_val)
    base_acc = np.mean(np.argmax(base_preds, axis=-1) == y_val) * 100.0

    turbo_preds, turbo_val_loss = turbo_engine.model.forward_and_loss(x_val, y_val)
    turbo_acc = np.mean(np.argmax(turbo_preds, axis=-1) == y_val) * 100.0

    print("\n" + "=" * 70)
    print("   RESULTS COMPARISON")
    print("=" * 70)
    print(f"| Metric                      | Baseline (Standard) | AcceleratorAI Turbo  |")
    print(f"|-----------------------------|---------------------|----------------------|")
    print(f"| Final Training Loss         | {base_losses[-1]:19.4f} | {turbo_losses[-1]:20.4f} |")
    print(f"| Validation Accuracy         | {base_acc:18.1f}% | {turbo_acc:19.1f}% |")
    print(f"| Total Training Time         | {base_time:18.2f}s | {turbo_time:19.2f}s |")
    print(f"| Injected Perturbations      |                   0 | {len(telemetry):20d} |")
    print(f"| Final Learning Torque (Nm)  |                 N/A | {telemetry[-1].learning_torque_nm:20.3f} |")
    print(f"| Peak Virtual RPM            |                 N/A | {max(t.rpm for t in telemetry):20.1f} |")
    print(f"| Wastegate Relief Events     |                   0 | {turbo_engine.wastegate.total_relief_events:20d} |")
    print("=" * 70)

    if turbo_acc >= base_acc:
        print("\n>>> ACCELERATION SUCCESSFUL: Turbo engine navigated complex manifold with superior convergence!")
    print("=" * 70)


if __name__ == "__main__":
    main()
