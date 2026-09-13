"""
Fine-Grained PyTorch Profiler & CUDA Event Timing Harness for AcceleratorAI.
Measures per-component latency (microseconds) and identifies CPU-GPU synchronization bottlenecks.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np

from accelerator_ai.benchmarks.nanogpt import NanoGPT, GPTConfig
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.models.torch_adapter import PyTorchTurbineWrapper


def create_benchmark_batch(batch_size: int, block_size: int, vocab_size: int, device: str):
    x = torch.randint(0, vocab_size, (batch_size, block_size), dtype=torch.long, device=device)
    y = torch.randint(0, vocab_size, (batch_size, block_size), dtype=torch.long, device=device)
    return x, y


def profile_pipeline_stages(engine: TurboLearningEngine, num_steps: int = 50, batch_size: int = 32, block_size: int = 128, device: str = "cuda:0"):
    vocab_size = 1000
    x, y = create_benchmark_batch(batch_size, block_size, vocab_size, device)

    # Warm-up (10 steps)
    print(f"[Profiler] Warming up GPU ({device}) for 10 steps...")
    for _ in range(10):
        engine.step(x, y)
    torch.cuda.synchronize()

    # CUDA Event Timers
    stages = {
        "step_total": [],
        "fwd_bwd": [],
    }

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"[Profiler] Measuring {num_steps} steps with high-resolution CUDA Events...")
    for _ in range(num_steps):
        start_event.record()
        res = engine.step(x, y)
        end_event.record()
        torch.cuda.synchronize()
        dt = start_event.elapsed_time(end_event)  # milliseconds
        stages["step_total"].append(dt)

    mean_total = float(np.mean(stages["step_total"]))
    std_total = float(np.std(stages["step_total"]))
    p50_total = float(np.percentile(stages["step_total"], 50))
    p95_total = float(np.percentile(stages["step_total"], 95))

    print("\n" + "=" * 65)
    print(" ACCELERATOR-AI CUDA EVENT TIMING REPORT")
    print("=" * 65)
    print(f"Total Steps Measured : {num_steps}")
    print(f"Mean Step Latency    : {mean_total:.3f} ms (±{std_total:.3f} ms)")
    print(f"Median (P50) Latency : {p50_total:.3f} ms")
    print(f"P95 Latency          : {p95_total:.3f} ms")
    print("=" * 65)
    return stages


def run_torch_profiler(engine: TurboLearningEngine, output_dir: str, batch_size: int = 32, block_size: int = 128, device: str = "cuda:0"):
    vocab_size = 1000
    x, y = create_benchmark_batch(batch_size, block_size, vocab_size, device)

    print(f"[Profiler] Running PyTorch Profiler with Chrome trace export...")
    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, "accelerator_profile_trace.json")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=10, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(15):
            engine.step(x, y)
            prof.step()

    print("\n" + "=" * 80)
    print(" TOP 15 OPERATORS BY CUDA TIME")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n" + "=" * 80)
    print(" TOP 15 OPERATORS BY CPU TIME (Identifies Host Stalls & Syncs)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    prof.export_chrome_trace(trace_path)
    print(f"\n[Profiler] Chrome trace exported to: {trace_path}")
    print("[Profiler] View interactively at chrome://tracing or speedscope.app")


def main():
    parser = argparse.ArgumentParser(description="AcceleratorAI High-Precision Profiler")
    parser.add_argument("--steps", type=int, default=50, help="Number of measurement steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Micro-batch size")
    parser.add_argument("--block-size", type=int, default=128, help="Sequence length")
    parser.add_argument("--profile", action="store_true", help="Run PyTorch Profiler and export Chrome trace")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results/profile", help="Trace directory")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    config = GPTConfig(vocab_size=1000, block_size=args.block_size, n_layer=4, n_head=4, n_embd=128)
    model = NanoGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    wrapper = PyTorchTurbineWrapper(model, optimizer, loss_fn=None)

    engine = TurboLearningEngine(
        model=wrapper,
        enable_default_injectors=False,
        enable_vvt=True,
        fault_tolerance_mode=False,
    )

    if args.profile:
        run_torch_profiler(engine, args.output_dir, args.batch_size, args.block_size, device)
    else:
        profile_pipeline_stages(engine, args.steps, args.batch_size, args.block_size, device)


if __name__ == "__main__":
    main()
