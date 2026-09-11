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

from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.core.shaft import DriveShaft
from accelerator_ai.core.metrics import (
    EngineTelemetry,
    calculate_pyrometer_temp,
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
from accelerator_ai.ecu.telemetry import TelemetryHub
from accelerator_ai.security.input_guard import InputGuard
from accelerator_ai.core.pipeline import (
    FluidPipeline,
    ExpressCoreRoundabout,
    AuxiliaryInjectionRoundabout,
    ResonantObservationRoundabout,
)

logger = logging.getLogger("accelerator_ai.engine")


class TurboLearningEngine:
    """
    Orchestrates the entire turbocharged AI learning cycle with physical shaft inertia,
    Braided DNA multi-strand control, Sequential Turbocharging (HP/LP), and Dynamic VVT.
    Powered by the Vibe Multi-Tier FluidPipeline.
    """

    def __init__(
        self,
        model: Any,
        base_learning_rate: float = 0.015,
        target_boost_psi: float = 14.7,
        shaft_inertia: float = 0.08,
        enable_default_injectors: bool = True,
        injectors: Optional[List[AsyncDataInjector]] = None,
        enable_sequential_turbo: bool = True,
        enable_vvt: bool = True,
        telemetry_interval: int = 1,
        fault_tolerance_mode: bool = True,
        input_guard: Optional[InputGuard] = None,
        distributed_coordinator: Optional[Any] = None,
    ):
        self.model = model
        self.current_step: int = 0
        self.current_epoch: int = 0
        self.previous_loss: float = 1.0
        self.fault_tolerance_mode = fault_tolerance_mode
        self.input_guard = input_guard or InputGuard()

        # Distributed Master-ECU Coordinator (DDP / FSDP lockstep sync)
        if distributed_coordinator is not None:
            self.distributed_coordinator = distributed_coordinator
        else:
            from accelerator_ai.ecu.distributed import DistributedECUCoordinator
            self.distributed_coordinator = DistributedECUCoordinator()

        # Physical Mechanical Drive Shaft
        self.shaft = DriveShaft(
            inertia=shaft_inertia,
            idle_rpm=800.0,
            friction_coeff=0.012,
        )

        # Core Turbines (Tier 0: Express Core Roundabout)
        self.intake = IntakeTurbine()
        self.filter = AirFilter()
        self.compressor = CompressorTurbine(shaft=self.shaft)
        self.intercooler = Intercooler()
        self.combustion = CombustionChamber()
        self.gradient_turbine = GradientTurbine(shaft=self.shaft, enable_twin_scroll=True)
        self.wastegate = WastegateValve()

        # Sequential Turbocharging (HP fast spool + LP compound boost)
        self.enable_sequential_turbo = enable_sequential_turbo
        self.sequential_turbo = SequentialTurboSystem() if enable_sequential_turbo else None

        # Variable Valve Timing (Dynamic cam phasing & micro-batch sizing)
        self.enable_vvt = enable_vvt
        self.vvt = VariableValveTiming(base_batch_size=32) if enable_vvt else None

        # Mount compressor and gradient turbine to common drive shaft
        self.compressor.attach_shaft(self.shaft)
        self.gradient_turbine.attach_shaft(self.shaft)

        # Braided DNA Helices Controller & Telemetry Hub
        self.braided_ecu = BraidedDNAController(base_learning_rate=base_learning_rate)
        self.telemetry_hub = TelemetryHub()

        # Asynchronous Multi-Point Injectors (Tier 1: Auxiliary Injection Ring)
        self.injectors: List[AsyncDataInjector] = []
        if injectors:
            self.injectors.extend(injectors)
        elif enable_default_injectors:
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
            telemetry_interval=telemetry_interval,
        )

    @property
    def virtual_rpm(self) -> float:
        """True physical shaft RPM."""
        return self.shaft.rpm

    def trigger_nos(self) -> None:
        """Triggers an immediate high-entropy chaos kick from the shock injector."""
        self.injection_roundabout.trigger_nos()

    def step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> CombustionResult:
        """
        Executes a single physically closed turbocharged learning cycle via FluidPipeline.
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
            )
            if not self.distributed_coordinator.is_master:
                self.observation_roundabout.vvt_locked_gear = sync_gear
                self.observation_roundabout.locked_learning_rate = sync_lr
                self.vvt.current_gear = sync_gear
                self.vvt.current_batch_size = self.vvt.gears[min(sync_gear - 1, len(self.vvt.gears) - 1)]
                self.braided_ecu.current_learning_rate = sync_lr

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

