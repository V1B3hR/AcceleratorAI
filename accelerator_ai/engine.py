"""
TurboLearningEngine: The high-performance engine orchestrator for AcceleratorAI.

Implements Mechanical Feedback Loop 1.0 + Braided Control:
Connects the physical DriveShaft, Compressor Wheel, Intercooler, Asynchronous
Injectors, Combustion Chamber, Gradient Turbine, Wastegate Valve, and
BraidedDNAController into an integrated, dynamically closed fluid-learning loop.
"""

from typing import List, Optional, Dict, Any, Tuple
import logging
import numpy as np

from accelerator_ai.core.flow_packet import FlowPacket, FlowPacketPool
from accelerator_ai.core.shaft import DriveShaft
from accelerator_ai.core.metrics import (
    EngineTelemetry,
    calculate_pyrometer_temp,
    calculate_learning_torque,
)
from accelerator_ai.turbines.intake import IntakeTurbine
from accelerator_ai.turbines.filter import AirFilter
from accelerator_ai.turbines.compressor import CompressorTurbine
from accelerator_ai.turbines.intercooler import Intercooler
from accelerator_ai.turbines.combustion import CombustionChamber, CombustionResult
from accelerator_ai.turbines.gradient_turbine import GradientTurbine
from accelerator_ai.turbines.wastegate import WastegateValve
from accelerator_ai.turbines.sequential_turbo import SequentialTurboSystem
from accelerator_ai.turbines.vvt import VariableValveTiming
from accelerator_ai.injectors.base_injector import AsyncDataInjector
from accelerator_ai.injectors.synthetic import SyntheticInjector
from accelerator_ai.injectors.realworld import RealWorldReservoirInjector
from accelerator_ai.injectors.shock import EntropyShockInjector
from accelerator_ai.ecu.braided_controller import BraidedDNAController
from accelerator_ai.ecu.telemetry import TelemetryHub, AsyncTelemetryHub
from accelerator_ai.security.input_guard import InputGuard
from accelerator_ai.config import EngineConfig
from accelerator_ai.core.pipeline import (
    FluidPipeline,
    ExpressCoreRoundabout,
    AuxiliaryInjectionRoundabout,
    ResonantObservationRoundabout,
)

logger = logging.getLogger("accelerator_ai.engine")


# Hierarchical Turbine Activation Levels
TURBO_LEVEL_CRUISE: int = 1      # Minimal path: zero-drag forward/backward/update
TURBO_LEVEL_REGULATED: int = 2   # Minimal path + fused on-device wastegate soft-clipping
TURBO_LEVEL_RESONANT: int = 3    # Regulated + VVT dynamic gearing + Braided DNA modulation
TURBO_LEVEL_FULL: int = 4        # Full FluidPipeline with 3 roundabout tiers & auxiliary injectors


class TurboLearningEngine:
    """
    Orchestrates the entire turbocharged AI learning cycle with physical shaft inertia,
    Braided DNA multi-strand control, Sequential Turbocharging (HP/LP), and Dynamic VVT.
    Powered by the Vibe Multi-Tier FluidPipeline.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[EngineConfig] = None,
        base_learning_rate: Optional[float] = None,
        target_boost_psi: Optional[float] = None,
        shaft_inertia: Optional[float] = None,
        enable_default_injectors: Optional[bool] = None,
        enable_sequential_turbo: Optional[bool] = None,
        enable_vvt: Optional[bool] = None,
        telemetry_interval: Optional[int] = None,
        fault_tolerance_mode: Optional[bool] = None,
        fast_physics: Optional[bool] = None,
        input_guard: Optional[InputGuard] = None,
        injectors: Optional[List[AsyncDataInjector]] = None,
        distributed_coordinator: Optional[Any] = None,
        enable_cuda_graph: Optional[bool] = None,
        adaptive_turbo: Optional[bool] = None,
        adaptive_check_interval: Optional[int] = None,
        enable_amp: Optional[bool] = None,
        amp_dtype: Optional[str] = None,
        enable_telemetry: Optional[bool] = None,
        async_telemetry: Optional[bool] = None,
        gradient_accumulation_steps: Optional[int] = None,
    ):
        self.model = model
        self.config = config or EngineConfig()

        # Resolve parameters from config or explicit overrides
        resolved_lr = base_learning_rate if base_learning_rate is not None else self.config.base_learning_rate
        resolved_boost = target_boost_psi if target_boost_psi is not None else self.config.target_boost_psi
        resolved_inertia = shaft_inertia if shaft_inertia is not None else self.config.shaft_inertia
        resolved_def_inj = enable_default_injectors if enable_default_injectors is not None else self.config.enable_default_injectors
        resolved_seq_turbo = enable_sequential_turbo if enable_sequential_turbo is not None else self.config.enable_sequential_turbo
        resolved_vvt = enable_vvt if enable_vvt is not None else self.config.enable_vvt
        resolved_telemetry = telemetry_interval if telemetry_interval is not None else self.config.telemetry_interval
        resolved_ft_mode = fault_tolerance_mode if fault_tolerance_mode is not None else self.config.fault_tolerance_mode
        resolved_fast_physics = fast_physics if fast_physics is not None else self.config.fast_physics
        resolved_cuda_graph = enable_cuda_graph if enable_cuda_graph is not None else self.config.enable_cuda_graph
        resolved_adaptive_turbo = adaptive_turbo if adaptive_turbo is not None else self.config.adaptive_turbo
        resolved_check_interval = adaptive_check_interval if adaptive_check_interval is not None else self.config.adaptive_check_interval
        resolved_enable_amp = enable_amp if enable_amp is not None else self.config.enable_amp
        resolved_amp_dtype = amp_dtype if amp_dtype is not None else self.config.amp_dtype
        resolved_enable_telemetry = enable_telemetry if enable_telemetry is not None else self.config.enable_telemetry
        resolved_async_telemetry = async_telemetry if async_telemetry is not None else self.config.async_telemetry
        resolved_auto_detect_gears = self.config.auto_detect_gears
        resolved_filter_cache = self.config.filter_cache_window
        resolved_hierarchical_turbo = self.config.hierarchical_turbo
        resolved_accum = gradient_accumulation_steps if gradient_accumulation_steps is not None else self.config.gradient_accumulation_steps

        self.current_step: int = 0
        self.current_epoch: int = 0
        self.previous_loss: float = 1.0
        self.fault_tolerance_mode = resolved_ft_mode
        self.fast_physics = resolved_fast_physics
        self.enable_cuda_graph = resolved_cuda_graph
        self.adaptive_turbo = resolved_adaptive_turbo
        self.adaptive_check_interval = resolved_check_interval
        self.hierarchical_turbo = resolved_hierarchical_turbo
        self.gradient_accumulation_steps = resolved_accum
        self.enable_amp = resolved_enable_amp
        self.amp_dtype = resolved_amp_dtype
        self.enable_telemetry = resolved_enable_telemetry
        self.async_telemetry = resolved_async_telemetry
        self.input_guard = input_guard or InputGuard()
        self.packet_pool = FlowPacketPool(max_size=64)
        self.loss_history: List[float] = []

        # CUDA Graph pre-allocated state
        self.cuda_graph: Optional[Any] = None
        self.static_x: Optional[Any] = None
        self.static_y: Optional[Any] = None
        self.static_predictions: Optional[Any] = None
        self.static_loss: Optional[Any] = None

        # Propagate AMP configuration to PyTorch wrapper if supported
        if hasattr(self.model, "enable_amp") and resolved_enable_amp:
            self.model.enable_amp = resolved_enable_amp
            self.model.amp_dtype = resolved_amp_dtype
            if (
                resolved_amp_dtype in ("float16", "fp16")
                and hasattr(self.model, "torch")
                and self.model.torch.cuda.is_available()
            ):
                self.model.scaler = self.model.torch.amp.GradScaler("cuda")

        # Distributed Master-ECU Coordinator (DDP / FSDP lockstep sync)
        if distributed_coordinator is not None:
            self.distributed_coordinator = distributed_coordinator
        else:
            from accelerator_ai.ecu.distributed import DistributedECUCoordinator
            self.distributed_coordinator = DistributedECUCoordinator()

        # Physical Mechanical Drive Shaft
        self.shaft = DriveShaft(
            inertia=resolved_inertia,
            idle_rpm=800.0,
            friction_coeff=0.012,
        )

        # Core Turbines (Tier 0: Express Core Roundabout)
        self.intake = IntakeTurbine()
        self.filter = AirFilter(cache_window=resolved_filter_cache)
        self.compressor = CompressorTurbine(shaft=self.shaft)
        if resolved_boost != 14.7:
            self.compressor.set_boost(1.0 + (resolved_boost / 14.7))
        self.intercooler = Intercooler()
        self.combustion = CombustionChamber()
        self.gradient_turbine = GradientTurbine(shaft=self.shaft, enable_twin_scroll=True)
        self.wastegate = WastegateValve()

        # Sequential Turbocharging (HP fast spool + LP compound boost)
        self.enable_sequential_turbo = resolved_seq_turbo
        self.sequential_turbo = SequentialTurboSystem() if resolved_seq_turbo else None

        # Variable Valve Timing (Dynamic cam phasing & micro-batch sizing)
        self.enable_vvt = resolved_vvt
        self.vvt = (
            VariableValveTiming(
                base_batch_size=self.config.gears[1] if len(self.config.gears) > 1 else 32,
                gears=None if resolved_auto_detect_gears else self.config.gears,
                auto_detect_hardware=resolved_auto_detect_gears,
            )
            if resolved_vvt
            else None
        )

        # Mount compressor and gradient turbine to common drive shaft
        self.compressor.attach_shaft(self.shaft)
        self.gradient_turbine.attach_shaft(self.shaft)

        # Braided DNA Helices Controller & Telemetry Hub
        self.braided_ecu = BraidedDNAController(base_learning_rate=resolved_lr)
        if resolved_async_telemetry:
            self.telemetry_hub = AsyncTelemetryHub(enabled=resolved_enable_telemetry)
        else:
            self.telemetry_hub = TelemetryHub(enabled=resolved_enable_telemetry)

        # Asynchronous Multi-Point Injectors (Tier 1: Auxiliary Injection Ring)
        self.injectors: List[AsyncDataInjector] = []
        if injectors:
            self.injectors.extend(injectors)
        elif resolved_def_inj:
            self.injectors.append(SyntheticInjector(batch_size=6))
            self.injectors.append(RealWorldReservoirInjector(batch_size=6))
            self.injectors.append(EntropyShockInjector(batch_size=4, threshold=0.85))

        # Reference to shock injector if present
        self.shock_injector: Optional[EntropyShockInjector] = next(
            (inj for inj in self.injectors if isinstance(inj, EntropyShockInjector)), None
        )

        # Assemble the 3-Tier Roundabout Manifold (Vibe Flow Pipeline)
        self.core_roundabout = ExpressCoreRoundabout(
            intake=self.intake,
            air_filter=self.filter,
            compressor=self.compressor,
            intercooler=self.intercooler,
            combustion=self.combustion,
            gradient_turbine=self.gradient_turbine,
            wastegate=self.wastegate,
        )
        self.injection_roundabout = AuxiliaryInjectionRoundabout(injectors=self.injectors)
        self.observation_roundabout = ResonantObservationRoundabout(
            shaft=self.shaft,
            braided_ecu=self.braided_ecu,
            sequential_turbo=self.sequential_turbo,
            vvt=self.vvt,
        )

        self.pipeline = FluidPipeline(
            core_roundabout=self.core_roundabout,
            injection_roundabout=self.injection_roundabout,
            observation_roundabout=self.observation_roundabout,
            telemetry_hub=self.telemetry_hub,
            telemetry_interval=resolved_telemetry,
        )

    @property
    def virtual_rpm(self) -> float:
        """True physical shaft RPM."""
        return self.shaft.rpm

    def trigger_nos(self) -> None:
        """Triggers an immediate high-entropy chaos kick from the shock injector."""
        self.injection_roundabout.trigger_nos()

    def capture_cuda_graph(self, x_batch: Any, y_batch: Any) -> None:
        """
        Captures the forward pass, backward pass, fused wastegate regulation,
        and optimizer update into a persistent torch.cuda.CUDAGraph.
        Eliminates CPU-GPU kernel launch and dispatch overhead.
        """
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch is not installed. Skipping CUDA Graph capture.")
            return

        if not torch.cuda.is_available():
            logger.warning("CUDA device not available. Skipping CUDA Graph capture.")
            return

        device = "cuda"
        if hasattr(self.model, "model") and hasattr(self.model.model, "parameters"):
            try:
                device = next(self.model.model.parameters()).device
            except StopIteration:
                pass

        if hasattr(self.model, "enable_capturable_optimizer"):
            self.model.enable_capturable_optimizer()

        clean_x, clean_y = self.input_guard.sanitize(x_batch, y_batch)
        if self.vvt:
            clean_x, clean_y = self.vvt.slice_batch(clean_x, clean_y)

        if not isinstance(clean_x, torch.Tensor):
            clean_x = torch.tensor(clean_x, device=device)
        else:
            clean_x = clean_x.to(device)

        if not isinstance(clean_y, torch.Tensor):
            clean_y = torch.tensor(clean_y, device=device)
        else:
            clean_y = clean_y.to(device)

        self.static_x = clean_x.clone().detach()
        self.static_y = clean_y.clone().detach()

        # Dedicated stream for capture to avoid legacy stream dependencies
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device=device))

        with torch.cuda.stream(capture_stream):
            # Warm up iterations (populates memory pool)
            for _ in range(3):
                self.model.forward_and_loss(self.static_x, self.static_y)
                if hasattr(self.model, "harvest_and_regulate_fused"):
                    self.model.harvest_and_regulate_fused(
                        threshold=self.wastegate.max_gradient_norm,
                        boost_ratio=self.compressor.boost_ratio,
                        enable_soft_clipping=self.wastegate.enable_soft_clipping,
                    )
                else:
                    self.model.backward()
                self.model.apply_updates(learning_rate=self.braided_ecu.current_learning_rate)

        torch.cuda.current_stream(device=device).wait_stream(capture_stream)

        # Graph Capture
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph, stream=capture_stream):
            self.static_predictions, _ = self.model.forward_and_loss(
                self.static_x, self.static_y
            )
            if hasattr(self.model, "harvest_and_regulate_fused"):
                self.model.harvest_and_regulate_fused(
                    threshold=self.wastegate.max_gradient_norm,
                    boost_ratio=self.compressor.boost_ratio,
                    enable_soft_clipping=self.wastegate.enable_soft_clipping,
                )
            else:
                self.model.backward()
            self.model.apply_updates(learning_rate=self.braided_ecu.current_learning_rate)

        self.static_loss_tensor = getattr(self.model, "last_loss_tensor", None)

        logger.info(
            "Captured persistent CUDA Graph for TurboLearningEngine on %s (batch shape: %s)",
            device,
            tuple(self.static_x.shape),
        )

    def _get_turbo_level(self, current_step: int) -> int:
        """
        Determines the hierarchical turbine activation level (1 to 4):
          Level 1: Minimal Cruise (zero-drag forward/backward/update)
          Level 2: Regulated (fused on-device soft-clipping wastegate)
          Level 3: Resonant (VVT dynamic gearing + Braided DNA modulation)
          Level 4: Full Fluid Roundabout Pipeline (all 3 tiers + auxiliary injectors)
        """
        # 1. Early exploration / spool-up phase -> Full boost
        if current_step <= 50:
            return TURBO_LEVEL_FULL

        # 2. Periodic calibration check -> Full boost
        if current_step % self.adaptive_check_interval == 0:
            return TURBO_LEVEL_FULL

        # 3. Wastegate knock / explosive gradient relief active -> Full relief
        if self.wastegate.open_pct > 0.0:
            return TURBO_LEVEL_FULL

        # 4. Learning stall / loss plateau detection -> Full boost
        if len(self.loss_history) >= 10:
            recent_delta = self.loss_history[-10] - self.previous_loss
            if recent_delta < 0.001:
                return TURBO_LEVEL_FULL

        # 5. Non-linear resonance modulation check
        if current_step % 3 == 0:
            return TURBO_LEVEL_RESONANT

        # 6. Smooth cruising
        return TURBO_LEVEL_CRUISE

    def _should_activate_turbo(self, current_step: int) -> bool:
        """
        Maintains backwards compatibility for full turbine pipeline activation.
        """
        return self._get_turbo_level(current_step) >= TURBO_LEVEL_FULL

    def step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> CombustionResult:
        """
        Executes a single physically closed turbocharged learning cycle via FluidPipeline,
        Zero-Sync CUDA Graph replay, or Adaptive Turbo Cruising.
        Protected by InputGuard and FaultTolerance bypass mode.
        """
        self.current_step += 1
        clean_x, clean_y = self.input_guard.sanitize(x_batch, y_batch)

        # In distributed mode, synchronize engine gear and dynamics across ranks
        if self.distributed_coordinator.is_distributed and self.vvt:
            sync_gear, sync_lr, _, _ = self.distributed_coordinator.broadcast_engine_state(
                vvt_gear=self.vvt.current_gear,
                learning_rate=self.braided_ecu.current_learning_rate,
                wastegate_open=bool(self.wastegate.open_pct > 0.0),
                shock_fired=bool(self.shock_injector and self.shock_injector.last_fired_step == self.current_step),
                step=self.current_step,
            )
            if not self.distributed_coordinator.is_master:
                self.observation_roundabout.vvt_locked_gear = sync_gear
                self.observation_roundabout.locked_learning_rate = sync_lr
                self.vvt.current_gear = sync_gear
                self.vvt.current_batch_size = self.vvt.gears[min(sync_gear - 1, len(self.vvt.gears) - 1)]
                self.braided_ecu.current_learning_rate = sync_lr

        # 0. Zero-Sync CUDA Graph Replay (Peak Hardware Limit)
        if (
            self.cuda_graph is not None
            and self.static_x is not None
            and hasattr(clean_x, "shape")
            and clean_x.shape == self.static_x.shape
        ):
            self.static_x.copy_(clean_x)
            self.static_y.copy_(clean_y)
            self.cuda_graph.replay()

            if getattr(self, "static_loss_tensor", None) is not None:
                loss_val = float(self.static_loss_tensor.item())
            else:
                loss_val = float(self.previous_loss)
            self.previous_loss = loss_val
            self.loss_history.append(loss_val)

            # Background ECU kinetics on interval strides
            if self.current_step % self.adaptive_check_interval == 0:
                compressor_load = self.compressor.compute_reaction_load()
                self.observation_roundabout.step_rotational_physics(
                    learning_torque=0.5,
                    compressor_load=compressor_load,
                    dt=0.08,
                )
                self.observation_roundabout.weave_dna(
                    step=self.current_step,
                    learning_torque=0.5,
                    boost_ratio=self.compressor.boost_ratio,
                    injected_entropy=0.0,
                    loss=loss_val,
                )

            fused_packet = self.packet_pool.acquire(
                x=clean_x,
                y=clean_y,
                pressure=self.compressor.boost_ratio,
            )
            return CombustionResult(
                loss=loss_val,
                predictions=self.static_predictions,
                fused_packet=fused_packet,
                exhaust_energy=float(loss_val * self.compressor.boost_ratio),
                air_fuel_ratio=14.7,
                homogeneity_pct=100.0,
            )

        if self.fast_physics:
            # High-throughput Fast-Physics Execution (Zero intermediate memory allocations)
            # 1. Discrete VVT gearbox micro-batching
            if self.vvt:
                clean_x, clean_y = self.vvt.slice_batch(clean_x, clean_y)

            # 2. Forward pass & loss
            predictions, loss = self.model.forward_and_loss(clean_x, clean_y)

            # 3. Exhaust harvesting & soft-clipping
            if hasattr(self.model, "harvest_and_regulate_fused"):
                grad_norm, clipped_norm, was_vented = self.model.harvest_and_regulate_fused(
                    threshold=self.wastegate.max_gradient_norm,
                    boost_ratio=self.compressor.boost_ratio,
                    enable_soft_clipping=self.wastegate.enable_soft_clipping,
                )
                raw_torque = calculate_learning_torque(grad_norm, self.compressor.boost_ratio)
                learning_torque = float(raw_torque * self.gradient_turbine.shaft_efficiency)
            else:
                fused_packet = FlowPacket(x=clean_x, y=clean_y, pressure=self.compressor.boost_ratio)
                grad_norm, learning_torque = self.gradient_turbine.harvest_gradients(
                    model=self.model,
                    fused_packet=fused_packet,
                    boost_ratio=self.compressor.boost_ratio,
                )
                clipped_norm, was_vented = self.wastegate.inspect_and_regulate(
                    model=self.model,
                    gradient_norm=grad_norm,
                    boost_ratio=self.compressor.boost_ratio,
                )

            # 4. Drive shaft Newton kinetics & Braided DNA
            compressor_load = self.compressor.compute_reaction_load()
            self.observation_roundabout.step_rotational_physics(
                learning_torque=learning_torque,
                compressor_load=compressor_load,
                dt=0.08,
            )
            braid_status, pyrometer_temp = self.observation_roundabout.weave_dna(
                step=self.current_step,
                learning_torque=learning_torque,
                boost_ratio=self.compressor.boost_ratio,
                injected_entropy=0.0,
                loss=loss,
            )

            # 5. Parameter update
            self.model.apply_updates(learning_rate=braid_status["learning_rate"])
            loss_val = float(loss)
            self.previous_loss = loss_val
            self.loss_history.append(loss_val)

            fused_packet = self.packet_pool.acquire(
                x=clean_x,
                y=clean_y,
                pressure=self.compressor.boost_ratio,
            )
            return CombustionResult(
                loss=loss_val,
                predictions=predictions,
                fused_packet=fused_packet,
                exhaust_energy=float(loss_val * self.compressor.boost_ratio),
                air_fuel_ratio=14.7,
                homogeneity_pct=100.0,
            )

        # Adaptive Turbo Cruising / Hierarchical Execution
        if self.adaptive_turbo and not self._should_activate_turbo(self.current_step):
            turbo_level = self._get_turbo_level(self.current_step) if self.hierarchical_turbo else TURBO_LEVEL_CRUISE

            if self.vvt:
                clean_x, clean_y = self.vvt.slice_batch(clean_x, clean_y)

            predictions, loss = self.model.forward_and_loss(clean_x, clean_y)

            # Level 2+: Fused On-Device Wastegate Regulation
            if turbo_level >= TURBO_LEVEL_REGULATED and hasattr(self.model, "harvest_and_regulate_fused"):
                grad_norm, clipped_norm, was_vented = self.model.harvest_and_regulate_fused(
                    threshold=self.wastegate.max_gradient_norm,
                    boost_ratio=self.compressor.boost_ratio,
                    enable_soft_clipping=self.wastegate.enable_soft_clipping,
                )
                if was_vented:
                    self.wastegate.open_pct = 50.0
                else:
                    self.wastegate.open_pct = max(0.0, self.wastegate.open_pct * 0.8)
            else:
                self.model.backward()
                self.wastegate.open_pct = max(0.0, self.wastegate.open_pct * 0.8)

            # Level 3+: Resonant DNA & Rotational Kinetics Modulation
            lr_to_apply = self.braided_ecu.current_learning_rate
            if turbo_level >= TURBO_LEVEL_RESONANT:
                compressor_load = self.compressor.compute_reaction_load()
                self.observation_roundabout.step_rotational_physics(
                    learning_torque=0.5,
                    compressor_load=compressor_load,
                    dt=0.08,
                )
                braid_status, _ = self.observation_roundabout.weave_dna(
                    step=self.current_step,
                    learning_torque=0.5,
                    boost_ratio=self.compressor.boost_ratio,
                    injected_entropy=0.0,
                    loss=float(loss),
                )
                lr_to_apply = braid_status["learning_rate"]

            self.model.apply_updates(learning_rate=lr_to_apply)

            loss_val = float(loss)
            self.previous_loss = loss_val
            self.loss_history.append(loss_val)

            fused_packet = self.packet_pool.acquire(
                x=clean_x,
                y=clean_y,
                pressure=self.compressor.boost_ratio,
            )
            return CombustionResult(
                loss=loss_val,
                predictions=predictions,
                fused_packet=fused_packet,
                exhaust_energy=float(loss_val),
                air_fuel_ratio=14.7,
                homogeneity_pct=100.0,
            )

        try:
            res = self.pipeline.flow_step(
                x_batch=clean_x,
                y_batch=clean_y,
                model=self.model,
                step_index=self.current_step,
                epoch_index=self.current_epoch,
            )
            self.previous_loss = res.loss
            return res
        except Exception as e:
            is_oom = "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower()
            if is_oom:
                logger.warning(
                    "TurboLearningEngine step %d encountered OutOfMemoryError! "
                    "Executing emergency CUDA memory flush and VVT downshift to Gear 1.",
                    self.current_step,
                )
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                if self.vvt:
                    self.vvt.current_gear = 1
                    self.vvt.current_batch_size = self.vvt.gears[0]
                    clean_x, clean_y = self.vvt.slice_batch(clean_x, clean_y)

                try:
                    predictions, loss = self.model.forward_and_loss(clean_x, clean_y)
                    self.model.backward()
                    self.model.apply_updates(learning_rate=self.braided_ecu.current_learning_rate)
                    self.previous_loss = float(loss)
                    fallback_packet = FlowPacket(x=clean_x, y=clean_y, pressure=1.0)
                    return CombustionResult(
                        loss=float(loss),
                        predictions=predictions,
                        fused_packet=fallback_packet,
                        exhaust_energy=float(loss),
                        air_fuel_ratio=14.7,
                        homogeneity_pct=100.0,
                    )
                except Exception as retry_err:
                    logger.error("Emergency micro-batch retry also failed: %s", retry_err)
                    if not self.fault_tolerance_mode:
                        raise

            if not self.fault_tolerance_mode:
                raise
            logger.warning(
                "TurboLearningEngine step %d encountered exception: %s. Executing graceful bypass step.",
                self.current_step,
                e,
                exc_info=False,
            )
            # Graceful degradation fallback: direct execution to keep cluster training alive
            predictions, loss = self.model.forward_and_loss(clean_x, clean_y)
            self.model.backward()
            self.model.apply_updates(learning_rate=self.braided_ecu.current_learning_rate)
            self.previous_loss = float(loss)
            fallback_packet = FlowPacket(x=clean_x, y=clean_y, pressure=1.0)
            return CombustionResult(
                loss=float(loss),
                predictions=predictions,
                fused_packet=fallback_packet,
                exhaust_energy=float(loss),
                air_fuel_ratio=14.7,
                homogeneity_pct=100.0,
            )

    def step_accumulated(
        self,
        x_micro: Any,
        y_micro: Any,
        step_in_cycle: int = 1,
        total_cycle_steps: Optional[int] = None,
    ) -> CombustionResult:
        """
        Executes a single micro-step within a multi-step gradient accumulation window.
        Gradients are scaled by (1 / total_cycle_steps), with intra-step wastegate
        inspection to prevent explosion. Model parameters update only on the
        final micro-step of the cycle.
        """
        total_k = total_cycle_steps or self.gradient_accumulation_steps
        is_final_step = (step_in_cycle >= total_k)
        is_first_step = (step_in_cycle == 1)

        clean_x, clean_y = self.input_guard.sanitize(x_micro, y_micro)
        if self.vvt:
            clean_x, clean_y = self.vvt.slice_batch(clean_x, clean_y)

        # Forward pass without clearing grads unless first step
        if hasattr(self.model, "forward_and_loss"):
            predictions, raw_loss = self.model.forward_and_loss(
                clean_x, clean_y, zero_grad=is_first_step
            )
            scaled_loss = raw_loss / float(total_k)
        else:
            predictions, raw_loss = self.model.forward_and_loss(clean_x, clean_y)
            scaled_loss = raw_loss / float(total_k)

        # Backward pass on scaled loss
        if hasattr(self.model, "scaler") and self.model.scaler is not None and self.model.scaler.is_enabled():
            self.model.scaler.scale(scaled_loss).backward()
        elif hasattr(scaled_loss, "backward"):
            scaled_loss.backward()
        else:
            self.model.backward()

        # On final accumulation micro-step: inspect/regulate accumulated gradients and apply updates
        grad_norm, was_vented = 0.0, False
        if is_final_step:
            self.current_step += 1
            if hasattr(self.model, "harvest_and_regulate_fused"):
                grad_norm, clipped_norm, was_vented = self.model.harvest_and_regulate_fused(
                    threshold=self.wastegate.max_gradient_norm,
                    boost_ratio=self.compressor.boost_ratio,
                    enable_soft_clipping=self.wastegate.enable_soft_clipping,
                    skip_backward=True,
                )
            lr = self.braided_ecu.current_learning_rate
            self.model.apply_updates(learning_rate=lr)

        loss_val = float(raw_loss)
        self.previous_loss = loss_val
        self.loss_history.append(loss_val)

        fused_packet = self.packet_pool.acquire(
            x=clean_x,
            y=clean_y,
            pressure=self.compressor.boost_ratio,
        )
        return CombustionResult(
            loss=loss_val,
            predictions=predictions,
            fused_packet=fused_packet,
            exhaust_energy=float(loss_val),
            air_fuel_ratio=14.7,
            homogeneity_pct=100.0,
        )

    def state_dict(self) -> Dict[str, Any]:
        """
        Serializes full engine physical dynamics, shaft kinetics, and ECU state.
        Allows seamless resumption from training checkpoints.
        """
        return {
            "version": "0.5.0",
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "previous_loss": float(self.previous_loss),
            "shaft": self.shaft.state_dict(),
            "braided_ecu": self.braided_ecu.state_dict(),
            "filter": self.filter.state_dict(),
            "vvt": self.vvt.state_dict() if self.vvt else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restores engine state from a checkpoint state_dict.
        """
        self.current_step = int(state_dict.get("current_step", 0))
        self.current_epoch = int(state_dict.get("current_epoch", 0))
        self.previous_loss = float(state_dict.get("previous_loss", 1.0))
        if "shaft" in state_dict and state_dict["shaft"] is not None:
            self.shaft.load_state_dict(state_dict["shaft"])
        if "braided_ecu" in state_dict and state_dict["braided_ecu"] is not None:
            self.braided_ecu.load_state_dict(state_dict["braided_ecu"])
        if "filter" in state_dict and state_dict["filter"] is not None:
            self.filter.load_state_dict(state_dict["filter"])
        if "vvt" in state_dict and self.vvt and state_dict["vvt"] is not None:
            self.vvt.load_state_dict(state_dict["vvt"])

    def train_epoch(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> List[float]:
        """Trains for one complete epoch across the dataset."""
        self.current_epoch += 1
        n = len(x_train)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        epoch_losses: List[float] = []
        for i in range(0, n, batch_size):
            batch_idx = indices[i : i + batch_size]
            res = self.step(x_train[batch_idx], y_train[batch_idx])
            epoch_losses.append(res.loss)

        return epoch_losses

