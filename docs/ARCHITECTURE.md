# AcceleratorAI: Architecture & Fluid Dynamics Specification

## 1. The Fluid Paradigm of Machine Learning

Traditional deep learning views training data as static discrete tensors indexed by batches $B \sim \mathcal{D}$.
**AcceleratorAI** replaces this view with a **continuous fluid mechanics model**:

$$\text{Data} \equiv \text{Compressible High-Velocity Fluid Medium}$$

The neural network training pipeline is re-engineered as a **turbocharged internal combustion engine** where information flows through aerodynamic ducts, undergoes compression, is enriched with asynchronous fuel injections, ignites in forward-loss combustion, and drives an exhaust turbine that feeds mechanical update torque back into the weights.

---

## 2. Core Equations

### 2.1 The FlowPacket Medium
A data stream is encapsulated as a `FlowPacket`:
- **Information Pressure ($\Psi$)**: Represents sample density and curriculum difficulty:
  $$\Psi = \Psi_{\text{manifold}} \cdot (1 + \text{HardExampleWeight})$$
- **Viscosity ($\eta$)**: Represents topological complexity and curvature of the feature manifold:
  $$Q = \frac{\Psi}{\eta} \quad \text{(Flow Throughput)}$$
- **Entropy Temperature ($T$)**: Variance and stochastic perturbation level.
- **Phase Angle ($\phi$)**: Temporal position along the engine rotation cycle.

### 2.2 Mechanical Drive Shaft Dynamics (Feedback Loop 1.0)
The exhaust turbine and intake compressor are physically linked via a common **Drive Shaft** with rotational moment of inertia $I$:

$$I \frac{d\omega}{dt} = \tau_{\text{gradient}} - \tau_{\text{compressor\_load}} - \beta (\omega - \omega_{\text{idle}})$$

where:
- $\tau_{\text{gradient}} = \|\nabla_\theta \mathcal{L}\| \times \Psi_{\text{boost}} \times \eta_{\text{shaft}}$ (Driving work harvested from backprop).
- $\tau_{\text{compressor\_load}} = C_L \cdot (\Psi_{\text{boost}} - 1.0) \cdot \left(\frac{\omega}{1000}\right)$ (Reaction load required to compress the fluid).
- $\beta$ is the viscous bearing drag coefficient.

Stored Rotational Kinetic Energy:
$$E_k = \frac{1}{2} I \omega^2$$

### 2.3 Turbomachinery Euler Pressure Rise
Centrifugal boost pressure ratio $\Psi_{\text{boost}}$ is a direct quadratic function of blade tip peripheral velocity:

$$\Psi_{\text{boost}} = 1.0 + \kappa \cdot \left(\frac{\omega}{\omega_{\text{idle}}}\right)^{1.5}$$

### 2.4 The DNA Plecionka (Braided Multi-Strand Dynamics)
Rather than a top-down scalar PID loop, the system is governed by **4 interwoven physical strands**:
1. **$\mathcal{S}_{\text{grad}}$ (Gradient Strand)**: $\tau(t) = \|\nabla \mathcal{L}\| \cdot \text{Boost}$.
2. **$\mathcal{S}_{\text{press}}$ (Pressure Strand)**: $\Psi(t)$, dictating hard-sample selection and augmentation magnitude.
3. **$\mathcal{S}_{\text{inj}}$ (Injection Strand)**: $\sum_k w_k \cdot \sin(\omega_k t + \Delta \phi_k + \epsilon)$.
4. **$\mathcal{S}_{\text{therm}}$ (Thermal Strand)**: $T(t)$, pyrometer heat driving intercooler damping and wastegate venting.

**Cross-Strand Phase Interference Matrix**:
$$M_{ij}(t) = \cos(\theta_i(t) - \theta_j(t))$$

**Helical Resonance Index ($\mathcal{H} \in [-1, 1]$)**:
$$\mathcal{H}(t) = \frac{1}{6} \sum_{i < j} M_{ij}(t)$$

- **$\mathcal{H} > 0$ (Constructive Resonance)**: Strands are in harmonic alignment; energy flows efficiently from gradient to compression; learning rate accelerates:
  $$\eta_{\text{braid}} = \eta_{\text{base}} \cdot (1.0 + 0.35 \cdot \mathcal{H}) \cdot \sqrt{\Psi_{\text{boost}}}$$
- **$\mathcal{H} < 0$ (Destructive Tension / Creative Bifurcation)**: Strands are desynchronized, preventing premature convergence. When prolonged tension occurs ($\bar{\mathcal{H}} < -0.25$), a phase symmetry-breaking event ("kopniak z boku") is fired to eject the model into an emergent basin.

---

## 3. Asynchronous Multi-Point Injection System

Traditional mini-batching feeds identical, synchronized batches in a rigid cadence. This monotony frequently traps models in suboptimal local basins or creates brittle representations.

AcceleratorAI introduces **asynchronous, phase-shifted fuel injectors**:

```
Injector 1 (Synthetic):     ───/\───/\───────/\──────/\───  (Δφ = 0.50 rad)
Injector 2 (Real-World):    ─────/\─────/\─────/\─────/\──  (Δφ = 1.80 rad)
Injector 3 (Chaos Shock):   ───────▲───────────────▲──────  (Δφ = 3.14 rad, Plateau Triggered)
Main Intake Charge:         ──══════════════════════════──  (Continuous Pressurized Stream)
                                            │
                                            ▼
                               [ COMBUSTION CHAMBER ]
```

### 3.1 Non-Linear Phase Interference
Each injector operates on an independent trigonometric phase wave:

$$\text{Pulse Trigger} \iff \sin(\omega_k \cdot t + \Delta \phi_k + \epsilon_{\text{jitter}}) \ge \Theta_k$$

Because frequencies $\omega_k$ and phase shifts $\Delta \phi_k$ are mutually prime or incommensurate, their interference pattern never repeats periodically.

### 3.2 Chaos Shock ("Kopniak z boku")
The `EntropyShockInjector` monitors the moving standard deviation of training loss:

$$\sigma_{\mathcal{L}}(K) = \sqrt{\frac{1}{K} \sum_{i=0}^{K-1} (\mathcal{L}_{t-i} - \bar{\mathcal{L}})^2}$$

When $\sigma_{\mathcal{L}}(K) < \epsilon_{\text{plateau}}$, the engine recognizes that the optimization trajectory is stuck in a plateau or flat saddle point. It immediately fires an orthogonal, high-entropy perturbation packet ("kopniak z boku"). This shock destabilizes the rigid weights just enough to eject the parameter vector out of the local minimum basin into a steeper descent trajectory.

---

## 4. Electronic Control Unit (ECU)

The `BoostController` executes a PID feedback loop:

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}$$

where error $e(t) = \mathcal{L}_{t-1} - \mathcal{L}_t$ (loss reduction rate).

When the exhaust pyrometer detects excessive heat ($T > 750^\circ\text{C}$, signifying exploding variance or imminent overfitting), the ECU instantly overrides boost and throttles the learning rate, stabilizing the training process.

---

## 5. Pneumatic Soft-Clipping Wastegate

Traditional gradient clipping clamps vectors with a non-differentiable step function:

$$g_{\text{hard}} = g \cdot \min\left(1, \frac{\text{threshold}}{\|g\|}\right)$$

This hard discontinuity causes gradient shockwaves that destabilize Transformer self-attention layers and produce high-frequency ringing in deep manifolds.

**AcceleratorAI** replaces this with continuous **pneumatic soft-clipping**:

$$\tilde{g} = g \cdot \tanh\left(\frac{\text{threshold}}{\|g\|}\right)$$

* When $\|g\| \ll \text{threshold}$, $\tanh(x) \approx x$, preserving small natural gradient steps identically.
* When $\|g\| \gg \text{threshold}$, the hyperbolic tangent smoothly bleeds over-pressure asymptotically toward the threshold, guaranteeing continuous differentiability $C^\infty$ across all parameter tensors.

---

## 6. $O(B \log B)$ Ultrasonic Filtering with Information Pressure ($\Psi$) Coupling

Standard duplicate and outlier detection suffers from quadratic $O(B^2)$ pairwise distance calculations, making it unusable for large batch LLM pre-training.

The **AirFilter** implements a two-tier screening strategy:
1. **Direct Pairwise Matrix**: When $B \le 64$, vectorized cosine similarity runs in a single BLAS GEMM call.
2. **Strided Random Projection Screening**: When $B > 64$, input features are projected into a 1D random manifold and sorted in $O(B \log B)$ time. Candidate duplicate pairs are screened only within a localized sliding window ($\text{stride} = 8$), eliminating quadratic memory spikes.
3. **Information Pressure ($\Psi$) Coupling**:
   When packet pressure $\Psi > 1.0$ (indicating dense, high-entropy tokens or hard examples), the ultrasonic dispersion threshold automatically relaxes from $\tau = 0.98$ to $\tau = 0.999$. This guarantees that rare, high-leverage support vectors are never aggressively filtered.

---

## 7. Variable Valve Timing (VVT) Discrete Gearbox & Zero-Copy Slicing

Continuous batch-size adjustments cause repeated CUDA buffer re-allocations and trigger JIT graph recompilations in `torch.compile` and CUDA Graphs.

The **Variable Valve Timing (VVT)** module implements a **discrete 3-gear physical transmission** ("Skrzynia Biegów"):
* **Gear 1 (16 samples)**: Engaged during startup and low-RPM spooling ($< 1400\text{ RPM}$). Generates higher gradient variance to rapidly discover dominant descent trajectories.
* **Gear 2 (32 samples)**: Cruising phase for standard balanced training dynamics.
* **Gear 3 (64 samples)**: Peak Boost / VTEC phase ($> 3000\text{ RPM}$, $\eta_v \ge 0.85$). Maximum stochastic averaging for fine-tuning deep basins.

**Zero-Copy Contiguous Slicing**:
All batch adjustments utilize slice views (`x[:target_n], y[:target_n]`), guaranteeing zero tensor allocations and full CUDA Graph / PyTorch 2.x JIT stability.

---

## 8. Distributed Master-ECU Synchronization (DDP / FSDP Lockstep)

In multi-GPU cluster training (PyTorch DistributedDataParallel or FSDP), independent per-GPU dynamics would desynchronize batch shapes and cause fatal NCCL `all_reduce` deadlocks.

The **`DistributedECUCoordinator`** establishes lockstep cluster execution:
1. **Rank 0 (Master ECU)**: Executes the thermodynamic physical loop and computes all control decisions: VVT gear selection, Braided DNA learning rate, wastegate venting, and shock injection status.
2. **Lockstep Broadcast**: Packs the 4 control variables into a 4-float tensor `[vvt_gear, lr, wastegate_flag, shock_flag]` and broadcasts across all ranks in a single collective call (`dist.broadcast`).
3. **Worker Ranks (Slaves)**: Lock local VVT batch slicing and optimizer learning rates to the master's broadcast state, guaranteeing identical tensor dimensions across every GPU prior to gradient reduction.
4. **Standalone Mode**: In single-GPU or non-distributed environments, the coordinator operates with zero overhead (`is_distributed = False`).

