# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-09-18

### Added
- **Closed-Loop Kalman Governor (`KalmanLossGovernor`)**: 2-state discrete-time state-space Kalman filter tracking latent loss and velocity $x = [\mathcal{L}, \dot{\mathcal{L}}]^T$ with stochastic noise rejection and continuous, jitter-free boost modulation for plateaus ($1.5\times$) and divergence ($0.7\times$).
- **Per-Module Circuit Breakers (`ModuleCircuitBreaker`)**: Enterprise fault isolation implementing the Netflix Hystrix pattern (`CLOSED` $\to$ `OPEN` $\to$ `HALF_OPEN`) on each individual turbine module to bypass localized runtime failures without stopping training.
- **Preemptive VRAM Pressure Guard (`VRAMPressureGuard`)**: Asynchronous device memory headroom monitoring via `torch.cuda.mem_get_info()`, recommending proactive VVT downshifting and cache purging before Out-Of-Memory exceptions can occur.
- **Atomic Single Source of Truth (`EngineState`)**: Unified state tracking kinetics, VVT gears, Kalman loss dynamics, VRAM pressure levels, tripped module breakers, and real-time step profiling.
- **Continuous Profiling & Overhead Tracking**: Real-time measurement of `engine_overhead_ms` and `compute_efficiency_pct` embedded directly in `EngineTelemetry`.
- Expanded test suite to **111 automated unit tests** with 100% pass rate.

## [0.5.0] - 2026-09-15

### Added
- **JIT FluidPipeline Compilation**: Leveraged `torch.compile` with `reduce-overhead` mode and zero-copy tensor views for accelerated graph execution.
- **CUDA Graphs Integration**: Added `CUDAGraphManager` for static-shape training steps, eliminating GPU launch latency overhead.
- **Biomimetic Turbine Enhancements**:
  - Sequential Twin-Scroll Turbocharger system for multi-stage gradient pressure distribution.
  - Variable Valve Timing (VVT) actuation for dynamic phase adjustments during loss plateaus.
  - Automated Adaptive Turbo boosting responding directly to loss curvature and variance.
- **Hardened Live Cockpit Server**:
  - Secure localhost-first binding (`127.0.0.1`).
  - Token-based authentication (`X-Cockpit-Token`, `Authorization: Bearer`, and authenticated query parameter).
  - Strict localhost CORS policies and input parameter boundary clamping.
  - Masked authentication tokens in console outputs, delegating sensitive URLs to logger debug output.
- **Plug-and-Play Ecosystem Integrations**:
  - Added `accelerator_ai.wrap()` 1-liner function for instant wrapping of PyTorch models and optimizers.
  - Added `AcceleratorAICallback` for Hugging Face `transformers.Trainer`.
  - Added `AcceleratorAILightningCallback` for PyTorch Lightning.
- **Multi-Epoch Benchmark Suite**:
  - Added `benchmarks/bench_long_epochs.py` validating convergence, dynamic shaft RPM spooling, and memory leak resistance across extended epochs.
- **Enterprise Release & Documentation**:
  - Automated PyPI publishing and release pipeline (`.github/workflows/release.yml`).
  - Interactive Material for MkDocs documentation portal configuration (`mkdocs.yml`).
  - Expanded unit test coverage to 98 test cases across 16 test modules.

### Changed
- Promoted development status classifier to `Development Status :: 5 - Production/Stable`.
- Specified strict upper bounds on runtime and optional dependencies (`numpy<3.0.0`, `torch<3.0.0`, `wandb<1.0.0`).
- Vectorized turbine telemetry collection to achieve near-zero metrics extraction latency.

## [0.4.0] - 2026-09-10

### Added
- Core turbine engine architecture (`TurboLearningEngine`, `FluidPipeline`, `CoreTurbineRoundabout`).
- Biomimetic injectors: Swirl, Intercooler, and Dispersion valves.
- Pure NumPy fallback model (`PureNumPyMLP`) and PyTorch bridge.
- Real-time SSE telemetry streaming protocol.

## [0.3.0] - 2026-08-20

### Added
- Initial experimental benchmark suite comparing vanilla PyTorch against biomimetic turbine optimizers.
- Diagnostic profiling scripts and telemetry metrics.

## [0.1.0] - 2026-07-01

### Added
- Initial conceptual prototype of biomimetic fluid dynamics applied to neural gradient optimization.
