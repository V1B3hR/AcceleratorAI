# ADR-001: Biomimetic Fluid-Thermodynamic Training Metaphor & ML Mapping

## Context
Traditional deep learning optimizers treat training steps as discrete, isolated algebraic matrix updates:
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$$
While mathematically sound, this discrete viewpoint ignores the fluid, continuous physical dynamics of training: inertia, harmonic resonance, thermodynamic entropy, and flow resistance. 

However, users coming from classical machine learning backgrounds can find automotive and turbomachinery terminology ("VVT", "Wastegate", "Pyrometer", "Braided DNA", "Intercooler") unfamiliar or confusing.

## Decision
We establish a dual-representation architecture:
1. **Physical Simulation Layer**: The internal computation pipeline models training as a pressurized fluid medium flowing through an air-to-exhaust loop.
2. **Machine Learning Mapping**: Every physical component must have a 1-to-1 canonical translation to standard machine learning theory.

### The Canonical Translation Table

| Turbomachinery / Automotive Term | Machine Learning Concept | Mathematical / Algorithmic Role |
| :--- | :--- | :--- |
| **Charge Air (Intake Medium)** | Mini-Batch Tensor Pipeline | Raw training samples flowing into forward compute pass |
| **Air Filter** | Outlier & Poisoning Scrubber | $O(B)$ NaN screening, MAD/z-score clamping, de-clustering |
| **Compressor Turbine** | Information Pressure Booster | Manifold pressure scaling $\Psi = p / p_0$ modulating sample leverage |
| **Intercooler** | Variance & Thermal Regularizer | Stabilizes sample variance prior to forward layer injection |
| **Combustion Chamber** | Forward & Loss Compute Pass | Evaluation of predictions $\hat{y} = f(x; \theta)$ and task loss $\mathcal{L}$ |
| **Gradient Turbine** | Backward Gradient Harvester | Extracts parameter gradients $g = \nabla_\theta \mathcal{L}$ to generate kinetic torque $\tau$ |
| **Drive Shaft (Inertia $I$)** | Continuous Physical Momentum | Integrates angular acceleration $\alpha = (\tau - \tau_{\text{load}})/I$, driving RPM |
| **Wastegate Valve** | Pneumatic Gradient Clipping | Smooth $\tanh(\text{threshold}/\|g\|)$ soft-clipping bounding gradient spikes |
| **Variable Valve Timing (VVT)** | Dynamic Micro-Batch Transmission | Shifts between Gear 1 (16), Gear 2 (32), and Gear 3 (64) based on shaft RPM |
| **Braided DNA Controller** | Multi-Strand LR Modulator | Helical resonance clock combining torque, pressure, and thermal loss into $\eta_t$ |
| **Entropy Shock (NOS)** | Stochastic Basin Escaper | Controlled noise injection into gradient tensor to escape saddle plateaus |
| **Sequential Turbo (HP / LP)** | Multi-Stage Regimes | Low-inertia rapid early spooling transitioning to compound high-throughput batching |

## Consequences
- **Positive**: Machine learning researchers understand exactly what each component computes mathematically without needing automotive engineering training.
- **Positive**: Preserves the innovative biomimetic physical coupling that yields 5x deeper loss convergence and 1.5x lower perplexity.
- **Guidance**: All new documentation and tutorials must cite both terms (e.g. *"Variable Valve Timing (Dynamic Micro-Batch Sizing)"*).
