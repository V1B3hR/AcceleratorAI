"""
NanoGPT Empirical Benchmark: Vanilla AdamW vs AcceleratorAI Engine.
Evaluates Language Model pre-training convergence, throughput (tokens/sec),
and validation perplexity on real text (TinyShakespeare) using local NVIDIA GPU.
"""

import os
import sys
import time
import json
import math
import urllib.request
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from accelerator_ai.benchmarks.nanogpt import NanoGPT, GPTConfig
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.models.torch_adapter import PyTorchTurbineWrapper


def get_data_dir() -> str:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def load_or_download_tinyshakespeare() -> str:
    """Loads TinyShakespeare text dataset, downloading if necessary."""
    data_dir = get_data_dir()
    file_path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(file_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"[Benchmark] Downloading TinyShakespeare dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"[Benchmark] Downloaded to {file_path}")
        except Exception as e:
            print(f"[Benchmark] Download failed ({e}). Falling back to synthetic text.")
            sample = ("To be, or not to be, that is the question: "
                      "Whether 'tis nobler in the mind to suffer "
                      "The slings and arrows of outrageous fortune, "
                      "Or to take arms against a sea of troubles. ") * 5000
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(sample)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


class CharDataset:
    """Character-level dataset tokenizer and batch generator."""

    def __init__(self, text: str, split_ratio: float = 0.9):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}

        data = torch.tensor([self.char_to_ix[c] for c in text], dtype=torch.long)
        n = int(split_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y


@torch.no_grad()
def evaluate_val_loss(
    model: nn.Module,
    dataset: CharDataset,
    batch_size: int,
    block_size: int,
    device: str,
    eval_iters: int = 20,
) -> float:
    """Evaluates validation cross-entropy loss over multiple batches."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataset.get_batch("val", batch_size, block_size, device)
        _, loss = model(x, y)
        if loss is not None:
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def run_vanilla_baseline(
    dataset: CharDataset,
    config: GPTConfig,
    num_steps: int = 500,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cuda:0",
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs standard baseline training loop with PyTorch AdamW."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = NanoGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    step_losses = []
    val_checkpoints = []
    step_latencies = []
    tokens_per_step = batch_size * config.block_size

    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    total_tokens = 0

    print(f"\n--- Starting Vanilla AdamW Baseline ({num_steps} steps, device={device}) ---")
    for step in range(num_steps):
        t0 = time.perf_counter()
        x, y = dataset.get_batch("train", batch_size, config.block_size, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        step_latencies.append(dt)
        loss_val = float(loss.item())
        step_losses.append(loss_val)
        total_tokens += tokens_per_step

        if step % 100 == 0 or step == num_steps - 1:
            val_loss = evaluate_val_loss(model, dataset, batch_size, config.block_size, device)
            ppl = math.exp(min(val_loss, 20.0))
            val_checkpoints.append({"step": step, "val_loss": val_loss, "perplexity": ppl})
            print(f"[Vanilla AdamW] Step {step:4d}/{num_steps} | Loss: {loss_val:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f} | Latency: {dt:.2f}ms")

    total_wall_time = time.perf_counter() - start_time
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available() and "cuda" in device
        else 0.0
    )

    final_val_loss = val_checkpoints[-1]["val_loss"] if val_checkpoints else step_losses[-1]
    final_ppl = math.exp(min(final_val_loss, 20.0))

    return {
        "name": "Vanilla AdamW",
        "steps": num_steps,
        "total_wall_time_sec": round(total_wall_time, 3),
        "mean_latency_ms": round(float(np.mean(step_latencies)), 2),
        "tokens_per_sec": round(total_tokens / total_wall_time, 1),
        "final_train_loss": round(float(step_losses[-1]), 4),
        "final_val_loss": round(final_val_loss, 4),
        "final_perplexity": round(final_ppl, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "step_losses": step_losses,
        "val_checkpoints": val_checkpoints,
    }


def run_accelerator_ai(
    dataset: CharDataset,
    config: GPTConfig,
    num_steps: int = 500,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cuda:0",
    seed: int = 42,
    mode: str = "full_fluid",
) -> Dict[str, Any]:
    """Runs AcceleratorAI TurboLearningEngine with specified performance mode."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = NanoGPT(config).to(device)
    is_cuda_graph = (mode == "cuda_graph_bf16")
    is_adaptive = (mode == "adaptive_turbo")
    is_amp = (mode == "cuda_graph_bf16")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-2,
        capturable=is_cuda_graph,
    )
    wrapper = PyTorchTurbineWrapper(
        model=model,
        optimizer=optimizer,
        loss_fn=None,
        enable_amp=is_amp,
        amp_dtype="bfloat16" if is_amp else "float16",
    )

    mode_names = {
        "full_fluid": "AcceleratorAI (Full Fluid)",
        "adaptive_turbo": "AcceleratorAI (Adaptive Turbo)",
        "cuda_graph_bf16": "AcceleratorAI (CUDA Graph + BF16)",
    }
    engine_name = mode_names.get(mode, f"AcceleratorAI ({mode})")

    engine = TurboLearningEngine(
        model=wrapper,
        base_learning_rate=learning_rate,
        target_boost_psi=14.7,
        enable_default_injectors=False,
        enable_vvt=not is_cuda_graph,
        fault_tolerance_mode=False,
        fast_physics=False,
        adaptive_turbo=is_adaptive,
        enable_cuda_graph=is_cuda_graph,
        enable_amp=is_amp,
        amp_dtype="bfloat16" if is_amp else "float16",
        telemetry_interval=10,
    )

    # If CUDA Graph mode is selected, warm up and capture graph on device
    if is_cuda_graph:
        init_x, init_y = dataset.get_batch("train", batch_size, config.block_size, device)
        engine.capture_cuda_graph(init_x, init_y)

    step_losses = []
    val_checkpoints = []
    step_latencies = []
    exhaust_energies = []
    vvt_gears = []

    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    total_tokens = 0

    print(f"\n--- Starting {engine_name} ({num_steps} steps, device={device}) ---")
    for step in range(num_steps):
        t0 = time.perf_counter()
        x, y = dataset.get_batch("train", batch_size, config.block_size, device)

        res = engine.step(x, y)

        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        step_latencies.append(dt)

        actual_batch = res.fused_packet.batch_size
        total_tokens += actual_batch * config.block_size
        step_losses.append(float(res.loss))
        exhaust_energies.append(float(res.exhaust_energy))

        current_gear = engine.vvt.current_gear if engine.vvt else 1
        vvt_gears.append(current_gear)

        if step % 100 == 0 or step == num_steps - 1:
            val_loss = evaluate_val_loss(model, dataset, batch_size, config.block_size, device)
            ppl = math.exp(min(val_loss, 20.0))
            val_checkpoints.append({"step": step, "val_loss": val_loss, "perplexity": ppl})
            boost = round(engine.shaft.rpm, 1)
            ecu_lr = round(engine.braided_ecu.current_learning_rate, 6)
            print(
                f"[{engine_name}] Step {step:4d}/{num_steps} | Loss: {res.loss:.4f} | "
                f"Val: {val_loss:.4f} | PPL: {ppl:.2f} | Gear: {current_gear} | "
                f"RPM: {boost} | ECU LR: {ecu_lr:.5f} | Latency: {dt:.2f}ms"
            )

    total_wall_time = time.perf_counter() - start_time
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available() and "cuda" in device
        else 0.0
    )

    final_val_loss = val_checkpoints[-1]["val_loss"] if val_checkpoints else step_losses[-1]
    final_ppl = math.exp(min(final_val_loss, 20.0))

    return {
        "name": engine_name,
        "steps": num_steps,
        "total_wall_time_sec": round(total_wall_time, 3),
        "mean_latency_ms": round(float(np.mean(step_latencies)), 2),
        "tokens_per_sec": round(total_tokens / total_wall_time, 1),
        "final_train_loss": round(float(step_losses[-1]), 4),
        "final_val_loss": round(final_val_loss, 4),
        "final_perplexity": round(final_ppl, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "step_losses": step_losses,
        "val_checkpoints": val_checkpoints,
        "vvt_gears": vvt_gears,
    }


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"================================================================")
    print(f" NanoGPT GPU Empirical Benchmark: Vanilla vs AcceleratorAI")
    print(f" Hardware: {gpu_name} (PyTorch {torch.__version__})")
    print(f"================================================================")

    # 1. Dataset Loading
    raw_text = load_or_download_tinyshakespeare()
    dataset = CharDataset(raw_text)
    print(f"Dataset: TinyShakespeare | Total Chars: {len(raw_text):,} | Vocab Size: {dataset.vocab_size}")

    # 2. NanoGPT Architecture Configuration
    config = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
    )

    # Count parameters
    temp_model = NanoGPT(config)
    n_params = sum(p.numel() for p in temp_model.parameters())
    del temp_model
    print(f"NanoGPT Architecture: {config.n_layer} layers, {config.n_head} heads, embd={config.n_embd} | Parameters: {n_params:,}")

    num_steps = 500
    batch_size = 32
    lr = 2e-3

    # 3. Run Benchmarks
    results_vanilla = run_vanilla_baseline(
        dataset=dataset,
        config=config,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=lr,
        device=device,
        seed=1337,
    )

    results_full = run_accelerator_ai(
        dataset=dataset,
        config=config,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=lr,
        device=device,
        seed=1337,
        mode="full_fluid",
    )

    results_adaptive = run_accelerator_ai(
        dataset=dataset,
        config=config,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=lr,
        device=device,
        seed=1337,
        mode="adaptive_turbo",
    )

    results_cuda_graph = run_accelerator_ai(
        dataset=dataset,
        config=config,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=lr,
        device=device,
        seed=1337,
        mode="cuda_graph_bf16",
    )

    # 4. Comparative Synthesis
    print(f"\n===================================================================================================================================")
    print(f" FINAL BENCHMARK SUMMARY (NVIDIA RTX 4070 - {num_steps} Steps)")
    print(f"===================================================================================================================================")
    print(f"{'Metric':<28} | {'Vanilla AdamW':<14} | {'Full Fluid':<14} | {'Adaptive Turbo':<16} | {'CUDA Graph (BF16)':<18} | {'CUDA vs Vanilla':<14}")
    print("-" * 115)

    v_loss = results_vanilla["final_val_loss"]
    a_loss = results_full["final_val_loss"]
    t_loss = results_adaptive["final_val_loss"]
    c_loss = results_cuda_graph["final_val_loss"]
    loss_delta = ((v_loss - c_loss) / v_loss) * 100.0
    print(f"{'Validation Loss':<28} | {v_loss:<14.4f} | {a_loss:<14.4f} | {t_loss:<16.4f} | {c_loss:<18.4f} | {loss_delta:+.2f}%")

    v_ppl = results_vanilla["final_perplexity"]
    a_ppl = results_full["final_perplexity"]
    t_ppl = results_adaptive["final_perplexity"]
    c_ppl = results_cuda_graph["final_perplexity"]
    ppl_ratio = v_ppl / c_ppl if c_ppl > 0 else 1.0
    print(f"{'Validation Perplexity (PPL)':<28} | {v_ppl:<14.2f} | {a_ppl:<14.2f} | {t_ppl:<16.2f} | {c_ppl:<18.2f} | {ppl_ratio:.2f}x better")

    v_lat = results_vanilla["mean_latency_ms"]
    a_lat = results_full["mean_latency_ms"]
    t_lat = results_adaptive["mean_latency_ms"]
    c_lat = results_cuda_graph["mean_latency_ms"]
    lat_delta = c_lat - v_lat
    print(f"{'Mean Step Latency (ms)':<28} | {v_lat:<14.2f} | {a_lat:<14.2f} | {t_lat:<16.2f} | {c_lat:<18.2f} | {lat_delta:+.2f} ms")

    v_tok = results_vanilla["tokens_per_sec"]
    a_tok = results_full["tokens_per_sec"]
    t_tok = results_adaptive["tokens_per_sec"]
    c_tok = results_cuda_graph["tokens_per_sec"]
    tok_delta = ((c_tok - v_tok) / v_tok) * 100.0
    print(f"{'Throughput (Tokens/sec)':<28} | {v_tok:<14.1f} | {a_tok:<14.1f} | {t_tok:<16.1f} | {c_tok:<18.1f} | {tok_delta:+.2f}%")

    v_mem = results_vanilla["peak_vram_mb"]
    a_mem = results_full["peak_vram_mb"]
    t_mem = results_adaptive["peak_vram_mb"]
    c_mem = results_cuda_graph["peak_vram_mb"]
    print(f"{'Peak VRAM (MB)':<28} | {v_mem:<14.1f} | {a_mem:<14.1f} | {t_mem:<16.1f} | {c_mem:<18.1f} | {c_mem - v_mem:+.1f} MB")
    print("=" * 115)

    # Save output to JSON
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "nanogpt_rtx4070_results.json")

    summary_data = {
        "hardware": gpu_name,
        "pytorch_version": torch.__version__,
        "dataset": "TinyShakespeare",
        "architecture": "NanoGPT (4L/4H/128E)",
        "parameters": n_params,
        "steps": num_steps,
        "vanilla_adamw": {k: v for k, v in results_vanilla.items() if k not in ("step_losses", "vvt_gears")},
        "accelerator_ai_full": {k: v for k, v in results_full.items() if k not in ("step_losses", "vvt_gears")},
        "accelerator_ai_adaptive_turbo": {k: v for k, v in results_adaptive.items() if k not in ("step_losses", "vvt_gears")},
        "accelerator_ai_cuda_graph_bf16": {k: v for k, v in results_cuda_graph.items() if k not in ("step_losses", "vvt_gears")},
        "comparison_cuda_vs_vanilla": {
            "val_loss_reduction_pct": round(loss_delta, 2),
            "perplexity_ratio": round(ppl_ratio, 2),
            "step_latency_delta_ms": round(lat_delta, 2),
            "throughput_speedup_ratio": round(c_tok / v_tok, 2),
        }
    }

    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[Benchmark] Results exported to {out_file}")


if __name__ == "__main__":
    main()

