"""
TurboLearningEngine: The high-performance engine orchestrator for AcceleratorAI.

Connects the Intake Turbine, Air Filter, Compressor, Intercooler,
Asynchronous Multi-Point Injectors, Combustion Chamber, Gradient Turbine,
Wastegate Valve, and ECU Boost Controller into an integrated fluid-dynamic training system.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from accelerator_ai.core.flow_packet import FlowPacket
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
from accelerator_ai.injectors.base_injector import AsyncDataInjector
from accelerator_ai.injectors.synthetic import SyntheticInjector
from accelerator_ai.injectors.realworld import RealWorldReservoirInjector
from accelerator_ai.injectors.shock import EntropyShockInjector
from accelerator_ai.ecu.controller import BoostController
from accelerator_ai.ecu.telemetry import TelemetryHub


class TurboLearningEngine:
    """
    Orchestrates the entire turbocharged AI learning cycle.
    """

    def __init__(
        self,
        model: Any,
        target_boost_psi: float = 12.0,
        base_learning_rate: float = 0.01,
        injectors: Optional[List[AsyncDataInjector]] = None,
        enable_default_injectors: bool = True,
    ):
        self.model = model
        self.current_step: int = 0
        self.current_epoch: int = 0
        self.previous_loss: float = 1.0
        self.virtual_rpm: float = 800.0  # Idle RPM

        # Core Turbines
        self.intake = IntakeTurbine()
        self.filter = AirFilter()
        self.compressor = CompressorTurbine(base_boost_ratio=1.0 + (target_boost_psi / 14.7))
        self.intercooler = Intercooler()
        self.combustion = CombustionChamber()
        self.gradient_turbine = GradientTurbine()
        self.wastegate = WastegateValve()

        # ECU & Telemetry Hub
        self.ecu = BoostController(
            target_boost_ratio=self.compressor.boost_ratio,
            base_learning_rate=base_learning_rate,
        )
        self.telemetry_hub = TelemetryHub()

        # Asynchronous Multi-Point Injectors
        self.injectors: List[AsyncDataInjector] = []
        if injectors:
            self.injectors.extend(injectors)
        elif enable_default_injectors:
            self.injectors.append(SyntheticInjector())
            self.injectors.append(RealWorldReservoirInjector())
            self.injectors.append(EntropyShockInjector())

        # Reference to shock injector if present
        self.shock_injector: Optional[EntropyShockInjector] = next(
            (inj for inj in self.injectors if isinstance(inj, EntropyShockInjector)), None
        )

    def trigger_nos(self) -> None:
        """Triggers an immediate high-entropy chaos kick from the shock injector."""
        if self.shock_injector:
            self.shock_injector.trigger_manual_shock()

    def step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> CombustionResult:
        """
        Executes a single turbocharged learning cycle.
        """
        self.current_step += 1

        # 1. Intake Stage: ingest and regulate laminar flow
        raw_packet = self.intake.ingest_raw(x_batch, y_batch)

        # 2. Air Filter Stage: clean out NaNs and outlier particles
        clean_packet = self.filter.process(raw_packet)

        # 3. Compressor Stage: boost information pressure Psi
        compressed_packet = self.compressor.process(clean_packet)

        # 4. Intercooler Stage: thermal normalization and cooling
        cooled_packet = self.intercooler.process(compressed_packet)

        # 5. Asynchronous Multi-Point Injections:
        # Check independent phase clocks and collect supplemental fuel packets
        injected_packets: List[FlowPacket] = []
        injected_entropy = 0.0

        for inj in self.injectors:
            pulse_packet = inj.pulse(self.current_step, context_packet=cooled_packet)
            if pulse_packet is not None:
                injected_packets.append(pulse_packet)
                injected_entropy += float(pulse_packet.temperature * pulse_packet.batch_size)

        # 6. Combustion Chamber: mix fuel, forward pass, ignite loss
        combustion_result = self.combustion.ignite(
            main_packet=cooled_packet,
            injected_packets=injected_packets,
            model=self.model,
        )
        loss = combustion_result.loss

        # Inform shock injector of loss for plateau detection
        if self.shock_injector:
            self.shock_injector.record_loss(loss)

        # 7. Gradient Turbine: exhaust backprop converts loss into Learning Torque
        grad_norm, learning_torque = self.gradient_turbine.harvest_gradients(
            model=self.model,
            fused_packet=combustion_result.fused_packet,
            boost_ratio=self.compressor.boost_ratio,
        )

        # 8. Wastegate Inspection: vent pressure and clip gradients if over-boost detected
        clipped_norm, was_vented = self.wastegate.inspect_and_regulate(
            model=self.model,
            gradient_norm=grad_norm,
            boost_ratio=self.compressor.boost_ratio,
        )

        # 9. Drive Shaft: apply weight updates using ECU-tuned learning rate
        self.model.apply_updates(learning_rate=self.ecu.learning_rate)

        # 10. Thermal & RPM calculation
        pyrometer_temp = calculate_pyrometer_temp(loss=loss)
        loss_improvement = max(0.0, self.previous_loss - loss)

        # Engine RPM dynamics: idle (800) + speed from torque and convergence rate
        target_rpm = 800.0 + (learning_torque * 450.0) + (loss_improvement * 5000.0)
        self.virtual_rpm = (0.75 * self.virtual_rpm) + (0.25 * target_rpm)

        # 11. ECU Update Cycle: PID modulation of boost and learning rate
        ecu_status = self.ecu.update(
            current_loss=loss,
            previous_loss=self.previous_loss,
            pyrometer_temp=pyrometer_temp,
            wastegate_open_pct=self.wastegate.open_pct,
        )
        self.compressor.set_boost(ecu_status["boost_ratio"])

        # Adapt injector dynamics based on whether loss improved
        reward = 1.0 if loss < self.previous_loss else -0.5
        for inj in self.injectors:
            inj.adapt_dynamics(reward)

        self.previous_loss = loss

        # 12. Telemetry Dispatch
        telemetry = EngineTelemetry(
            step=self.current_step,
            epoch=self.current_epoch,
            rpm=float(self.virtual_rpm),
            boost_psi=float(self.compressor.boost_psi),
            manifold_pressure=float(self.compressor.boost_ratio),
            pyrometer_temp_c=float(pyrometer_temp),
            loss=float(loss),
            learning_torque_nm=float(learning_torque),
            wastegate_open_pct=float(self.wastegate.open_pct),
            air_fuel_ratio=float(combustion_result.air_fuel_ratio),
            injected_entropy=float(injected_entropy),
            active_injectors=len(injected_packets),
            learning_rate=float(self.ecu.learning_rate),
        )
        self.telemetry_hub.emit(telemetry)

        return combustion_result

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
