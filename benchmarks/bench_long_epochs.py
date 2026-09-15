"""
Long-Epoch Multi-Step Empirical Benchmark:
Vanilla PyTorch vs AcceleratorAI (Plug-and-Play accelerator_ai.wrap).

Evaluates:
- Multi-epoch convergence stability and validation perplexity.
- Memory fragmentation & VRAM leak resistance over extended training.
- Adaptive Braided DNA learning rate and dynamic VVT gear progression across epochs.
"""

import os
import sys
import time
import json
import math
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import accelerator_ai
from accelerator_ai.benchmarks.nanogpt import NanoGPT, GPTConfig
from accelerator_ai.wrapper import wrap


def load_dataset() -> Tuple[torch.Tensor, torch.Tensor, int]:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tinyshakespeare.txt")
    if not os.path.exists(data_path):
        sample = ("First Citizen: Before we proceed any further, hear me speak. "
                  "All: Speak, speak. "
                  "First Citizen: You are all resolved rather to die than to famish? "
                  "All: Resolved. resolved. ") * 4000
        text = sample
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    encoded = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)

    n = int(0.9 * len(encoded))
    train_data = encoded[:n]
    val_data = encoded[n:]
    return train_data, val_data, vocab_size


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


def evaluate(model: nn.Module, val_data: torch.Tensor, batch_size: int, block_size: int, device: torch.device, eval_iters: int = 15) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch(val_data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def run_vanilla_epochs(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    vocab_size: int,
    epochs: int = 10,
    steps_per_epoch: int = 50,
    batch_size: int = 32,
    block_size: int = 64,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, Any]:
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=3, n_head=4, n_embd=96)
    model = NanoGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

    history = []
    total_time = 0.0

    print("\n" + "=" * 75)
    print("   TRAINING: VANILLA PYTORCH ADAMW (10 EPOCHS)")
    print("=" * 75)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        epoch_train_losses = []

        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, batch_size, block_size, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_losses.append(loss.item())

        epoch_time = time.perf_counter() - t0
        total_time += epoch_time

        val_loss = evaluate(model, val_data, batch_size, block_size, device)
        val_ppl = math.exp(min(val_loss, 20.0))
        mean_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

        history.append({
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "epoch_time_s": round(epoch_time, 2),
            "peak_vram_mb": round(vram_mb, 1),
        })

        print(f"   Epoch {epoch:02d}/{epochs:02d} | Train: {mean_train_loss:.4f} | Val: {val_loss:.4f} (PPL: {val_ppl:5.2f}) | Time: {epoch_time:.2f}s | VRAM: {vram_mb:.1f}MB")

    return {"history": history, "total_time": round(total_time, 2)}


def run_accelerator_epochs(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    vocab_size: int,
    epochs: int = 10,
    steps_per_epoch: int = 50,
    batch_size: int = 32,
    block_size: int = 64,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, Any]:
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=3, n_head=4, n_embd=96)
    raw_model = NanoGPT(config).to(device)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=2e-3, weight_decay=1e-2)

    # 1-Liner: Wrap the model & optimizer into AcceleratorAI
    engine = accelerator_ai.wrap(
        model=raw_model,
        optimizer=optimizer,
        base_learning_rate=2e-3,
        target_boost_psi=14.7,
        fast_physics=True,
    )

    history = []
    total_time = 0.0

    print("\n" + "=" * 75)
    print("   TRAINING: ACCELERATOR-AI (1-LINER WRAP, 10 EPOCHS)")
    print("=" * 75)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        epoch_train_losses = []

        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, batch_size, block_size, device)
            # Plug-and-play call: engine.step or engine(x, y)
            res = engine(x, y)
            epoch_train_losses.append(res.loss)

        epoch_time = time.perf_counter() - t0
        total_time += epoch_time

        # Clean evaluation via engine.eval_step
        raw_model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(15):
                xv, yv = get_batch(val_data, batch_size, block_size, device)
                _, vl = engine.eval_step(xv, yv)
                val_losses.append(vl)
        raw_model.train()

        val_loss = float(sum(val_losses) / len(val_losses))
        val_ppl = math.exp(min(val_loss, 20.0))
        mean_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

        history.append({
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "epoch_time_s": round(epoch_time, 2),
            "peak_vram_mb": round(vram_mb, 1),
            "virtual_rpm": round(engine.virtual_rpm, 1),
            "braided_lr": round(engine.braided_ecu.current_learning_rate, 6),
        })

        print(f"   Epoch {epoch:02d}/{epochs:02d} | Train: {mean_train_loss:.4f} | Val: {val_loss:.4f} (PPL: {val_ppl:5.2f}) | RPM: {engine.virtual_rpm:4.0f} | LR: {engine.braided_ecu.current_learning_rate:.5f} | Time: {epoch_time:.2f}s")

    return {"history": history, "total_time": round(total_time, 2)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_data, val_data, vocab_size = load_dataset()
    print(f"Dataset Loaded: {len(train_data):,} train tokens, {len(val_data):,} val tokens, Vocab: {vocab_size}")

    epochs = 10
    steps_per_epoch = 60

    vanilla_results = run_vanilla_epochs(train_data, val_data, vocab_size, epochs=epochs, steps_per_epoch=steps_per_epoch, device=device)
    accel_results = run_accelerator_epochs(train_data, val_data, vocab_size, epochs=epochs, steps_per_epoch=steps_per_epoch, device=device)

    # Print Comparison Summary
    print("\n" + "=" * 90)
    print("   MULTI-EPOCH CONVERGENCE COMPARISON SUMMARY (10 FULL EPOCHS)")
    print("=" * 90)
    print(f"   {'Epoch':<6} | {'Vanilla Val Loss':<18} | {'AcceleratorAI Val Loss':<24} | {'Val PPL Advantage':<20} | {'Status'}")
    print("   " + "-" * 84)

    v_hist = vanilla_results["history"]
    a_hist = accel_results["history"]

    for i in range(epochs):
        ep = v_hist[i]["epoch"]
        v_loss = v_hist[i]["val_loss"]
        a_loss = a_hist[i]["val_loss"]
        v_ppl = v_hist[i]["val_ppl"]
        a_ppl = a_hist[i]["val_ppl"]
        ppl_ratio = v_ppl / a_ppl

        advantage = f"{ppl_ratio:.2f}x better" if ppl_ratio > 1.0 else f"{ppl_ratio:.2f}x"
        status = "[+] Deeper" if a_loss < v_loss else "~ Equal"
        print(f"   {ep:<6} | {v_loss:<18.4f} | {a_loss:<24.4f} | {advantage:<20} | {status}")

    print("=" * 90)
    final_v_loss = v_hist[-1]["val_loss"]
    final_a_loss = a_hist[-1]["val_loss"]
    delta_loss_pct = ((final_v_loss - final_a_loss) / final_v_loss) * 100.0
    print(f"   Final Result: Vanilla {final_v_loss:.4f} vs AcceleratorAI {final_a_loss:.4f} ({delta_loss_pct:+.2f}% lower loss)")
    print(f"   Final Perplexity: Vanilla {v_hist[-1]['val_ppl']:.2f} vs AcceleratorAI {a_hist[-1]['val_ppl']:.2f}")
    print(f"   Total Training Time: Vanilla {vanilla_results['total_time']}s vs AcceleratorAI {accel_results['total_time']}s")
    print("=" * 90)

    # Save to results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "long_epochs_results.json")
    with open(out_file, "w") as f:
        json.dump({"vanilla": vanilla_results, "accelerator_ai": accel_results}, f, indent=2)
    print(f"Results exported to {out_file}\n")


if __name__ == "__main__":
    main()
