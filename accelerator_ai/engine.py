"""
TurboLearningEngine: The high-performance engine orchestrator for AcceleratorAI.

Implements Mechanical Feedback Loop 1.0 + Braided Control:
Connects the physical DriveShaft, Compressor Wheel, Intercooler, Asynchronous
Injectors, Combustion Chamber, Gradient Turbine, Wastegate Valve, and
BraidedDNAController into an integrated, dynamically closed fluid-learning loop.
"""

from typing import List, Optional, Dict, Any, Tuple
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
from accelerator_ai.injectors.base_injector import AsyncDataInjector
from accelerator_ai.injectors.synthetic import SyntheticInjector
from accelerator_ai.injectors.realworld import RealWorldReservoirInjector
from accelerator_ai.injectors.shock import EntropyShockInjector
from accelerator_ai.ecu.controller import BoostController
from accelerator_ai.ecu.braided_controller import BraidedDNAController
from accelerator_ai.ecu.telemetry import TelemetryHub


class TurboLearningEngine:
    """
    Orchestrates the entire turbocharged AI learning cycle with physical shaft inertia
    and Braided DNA multi-strand control.
    """

    def __init__(
        self,
        model: Any,
        base_learning_rate: float = 0.015,
        target_boost_psi: float = 14.7,
        shaft_inertia: float = 0.08,
        enable_default_injectors: bool = True,
        injectors: Optional[List[AsyncDataInjector]] = None,
    ):
        self.model = model
        self.current_step: int = 0
        self.current_epoch: int = 0
        self.previous_loss: float = 1.0

        # Physical Mechanical Drive Shaft
        self.shaft = DriveShaft(
            inertia=shaft_inertia,
            idle_rpm=800.0,
            friction_coeff=0.012,
        )

        # Core Turbines
        self.intake = IntakeTurbine()
        self.filter = AirFilter()
        self.compressor = CompressorTurbine(shaft=self.shaft)
        self.intercooler = Intercooler()
        self.combustion = CombustionChamber()
        self.gradient_turbine = GradientTurbine(shaft=self.shaft)
        self.wastegate = WastegateValve()

        # Mount compressor and gradient turbine to common drive shaft
        self.compressor.attach_shaft(self.shaft)
        self.gradient_turbine.attach_shaft(self.shaft)

        # Braided DNA Helices Controller & Telemetry Hub
        self.braided_ecu = BraidedDNAController(base_learning_rate=base_learning_rate)
        # Classic controller for backup telemetry/PID compatibility
        self.legacy_ecu = BoostController(base_learning_rate=base_learning_rate)
        self.telemetry_hub = TelemetryHub()

        # Asynchronous Multi-Point Injectors (Stoichiometric auxiliary fuel)
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

    @property
    def virtual_rpm(self) -> float:
        """True physical shaft RPM."""
        return self.shaft.rpm

    def trigger_nos(self) -> None:
        """Triggers an immediate high-entropy chaos kick from the shock injector."""
        if self.shock_injector:
            self.shock_injector.trigger_manual_shock()

    def step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> CombustionResult:
        """
        Executes a single physically closed turbocharged learning cycle.
        """
        self.current_step += 1

        # 1. Intake Stage: Ingest and regulate laminar flow
        raw_packet = self.intake.ingest_raw(x_batch, y_batch)

        # 2. Air Filter Stage: Clean out NaNs and outlier particles
        clean_packet = self.filter.process(raw_packet)

        # 3. Compressor Stage: Boost pressure ratio is directly driven by physical shaft RPM
        compressed_packet = self.compressor.process(clean_packet)

        # 4. Intercooler Stage: Thermal normalization and cooling
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

        # 6. Combustion Chamber: Mix fuel, forward pass, ignite loss
        combustion_result = self.combustion.ignite(
            main_packet=cooled_packet,
            injected_packets=injected_packets,
            model=self.model,
        )
        loss = combustion_result.loss

        # Inform shock injector of loss for plateau detection
        if self.shock_injector:
            self.shock_injector.record_loss(loss)

        # 7. Gradient Turbine: Harvest gradient norm and extract driving torque
        grad_norm, learning_torque = self.gradient_turbine.harvest_gradients(
            model=self.model,
            fused_packet=combustion_result.fused_packet,
            boost_ratio=self.compressor.boost_ratio,
        )

        # 8. Wastegate Inspection: Vent pressure and clip gradients if over-boost detected
        clipped_norm, was_vented = self.wastegate.inspect_and_regulate(
            model=self.model,
            gradient_norm=grad_norm,
            boost_ratio=self.compressor.boost_ratio,
        )

        # 9. Physical Drive Shaft Dynamic Integration:
        # Driving torque from Gradient Turbine accelerates shaft;
        # Reaction load from Compressor decelerates shaft;
        # Bearing drag dissipates energy.
        compressor_load = self.compressor.compute_reaction_load()
        self.shaft.step(
            torque_in=learning_torque,
            load_torque=compressor_load,
            dt=0.08,
        )

        # 10. Thermal & Pyrometer Calculation
        pyrometer_temp = calculate_pyrometer_temp(loss=loss)

        # 11. Braided DNA Helices Control Cycle:
        # Weaves Gradient Strand, Pressure Strand, Injection Strand, and Thermal Strand.
        braid_status = self.braided_ecu.update(
            step=self.current_step,
            learning_torque=learning_torque,
            boost_ratio=self.compressor.boost_ratio,
            injected_entropy=injected_entropy,
            pyrometer_temp=pyrometer_temp,
            loss=loss,
        )

        # If prolonged phase desynchronization tension occurred, trigger a phase symmetry break
        if braid_status["should_phase_shock"]:
            self.trigger_nos()

        # 12. Drive Shaft: Apply weight updates using Braided ECU-tuned learning rate
        self.model.apply_updates(learning_rate=braid_status["learning_rate"])

        # Adapt injector dynamics based on whether loss improved
        reward = 1.0 if loss < self.previous_loss else -0.5
        for inj in self.injectors:
            inj.adapt_dynamics(reward)

        self.previous_loss = loss

        # 13. Telemetry Dispatch
        telemetry = EngineTelemetry(
            step=self.current_step,
            epoch=self.current_epoch,
            rpm=float(self.shaft.rpm),
            boost_psi=float(self.compressor.boost_psi),
            manifold_pressure=float(self.compressor.boost_ratio),
            pyrometer_temp_c=float(pyrometer_temp),
            loss=float(loss),
            learning_torque_nm=float(learning_torque),
            compressor_load_nm=float(compressor_load),
            shaft_kinetic_energy_j=float(self.shaft.kinetic_energy),
            angular_accel_rad_s2=float(self.shaft.last_angular_accel),
            wastegate_open_pct=float(self.wastegate.open_pct),
            air_fuel_ratio=float(combustion_result.air_fuel_ratio),
            injected_entropy=float(injected_entropy),
            active_injectors=len(injected_packets),
            learning_rate=float(braid_status["learning_rate"]),
            helical_resonance=float(braid_status["resonance_index"]),
            phase_tension=float(braid_status["phase_tension"]),
            winding_number=float(braid_status["winding_number"]),
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
