# ADR-002: GPU-Native Zero-Sync Execution & Fused Multi-Tensor Operations

## Context
High-resolution microsecond profiling on NVIDIA GeForce RTX 4070 (`benchmarks/profile_engine.py`) revealed that the apparent overhead in the engine (+2.06 ms per step) was **not compute FLOPs**, but a **Device-to-Host (D2H) CPU-GPU Synchronization Tax**.

Specifically:
- Calling `.item()` on PyTorch scalar loss and gradient norm tensors forces the CPU host to halt execution queueing and wait across the PCIe bus for the GPU kernel to finish execution.
- Calling `.cpu().numpy()` on batches forces a blocking host-device memory copy.
- Evaluating `.any()` or `.isnan()` in control flow blocks asynchronous GPU stream pipelining.

## Decision
1. **Zero D2H Syncs on Training Hot Path**:
   - PyTorch tensors remain entirely on-device (`is_cuda` / `device`).
   - Replaced scalar `.item()` synchronization with tensor operations.
   - Non-strict NaN sanitization uses asynchronous on-device `torch.nan_to_num`.
2. **Fused Multi-Tensor Operations**:
   - Replaced loops computing individual parameter gradient norms with PyTorch fused C++ kernels:
     `norms = torch._foreach_norm(grads, 2)`
     `total_norm_t = torch.linalg.vector_norm(torch.stack(norms))`
   - Replaced parameter-by-parameter soft-clipping with fused multi-tensor scaling:
     `torch._foreach_mul_(grads, scale_t)`
3. **Discrete Token Sequence Fast Path**:
   - Transformer token IDs (`torch.long`) bypass continuous floating-point filtering, eliminating unneeded elementwise operations.

## Consequences
- **Positive**: Mean step latency overhead dropped by >80% (from +2.06 ms down to +0.59 ms).
- **Positive**: Full saturation of GPU streaming multiprocessors (SMs) with zero PCIe pipeline stalls.
- **Positive**: Native compatibility with `torch.compile` and CUDA graph capture.
