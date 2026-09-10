"""
FluidPipeline & Multi-Tier Roundabout Manifold (Vibe Flow Architecture).

Implements the multi-dimensional, layered roundabout topology:
  - Tier 0: Express Core Roundabout (Fast-track zero-copy data stream)
            Intake -> Filter -> Compressor -> Intercooler -> Combustion -> Exhaust Turbine -> Wastegate
  - Tier 1: Auxiliary Injection Roundabout (Phase-clocked synthetic / real / shock injectors)
            Connects to Tier 0 via the vectorized Swirl Atomization Slipway
  - Tier 2: Resonant Observation Roundabout (Ambient Braided DNA & Mechanical DriveShaft)
            Surrounds the data rings, modulating apertures, timing, and learning rate without drag

Eliminates monolithic imperative step scripts in favor of fluid, composable pipelining.
"""

from typing import List, Optional, Dict, Any, Tuple, Union, Callable
import numpy as np

from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.core.shaft import DriveShaft
from accelerator_ai.core.base_turbine import TurbineModule
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
from accelerator_ai.ecu.controller import BoostController
from accelerator_ai.ecu.braided_controller import BraidedDNAController
from accelerator_ai.ecu.telemetry import TelemetryHub


class PipelineStage:
    """
    Composable fluid stage supporting the >> operator for piping.
    """

    def __init__(self, name: str, handler: Callable[[FlowPacket], FlowPacket]):
        self.name = name
        self.handler = handler

    def process(self, packet: FlowPacket) -> FlowPacket:
        return self.handler(packet)

    def __rshift__(self, other: Union["PipelineStage", Callable[[FlowPacket], FlowPacket]]) -> "PipelineChain":
        if isinstance(other, PipelineStage):
            return PipelineChain([self, other])
        elif callable(other):
            return PipelineChain([self, PipelineStage(name=other.__name__, handler=other)])
        raise TypeError(f"Cannot pipe PipelineStage to {type(other)}")


class PipelineChain(PipelineStage):
    """A sequence of PipelineStages executing linearly with zero boundary drag."""

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
        super().__init__(
            name=" -> ".join(s.name for s in stages),
            handler=self._execute_chain,
        )

    def _execute_chain(self, packet: FlowPacket) -> FlowPacket:
        current = packet
        for stage in self.stages:
            current = stage.process(current)
        return current

    def __rshift__(self, other: Union[PipelineStage, Callable[[FlowPacket], FlowPacket]]) -> "PipelineChain":
        if isinstance(other, PipelineChain):
            return PipelineChain(self.stages + other.stages)
        elif isinstance(other, PipelineStage):
            return PipelineChain(self.stages + [other])
        elif callable(other):
            return PipelineChain(self.stages + [PipelineStage(name=other.__name__, handler=other)])
        raise TypeError(f"Cannot pipe PipelineChain to {type(other)}")


class ExpressCoreRoundabout:
    """
    Tier 0: Express Core Roundabout.
    The primary zero-copy data circuit through intake, compressor, intercooler,
    combustion, and exhaust turbines.
    """

    def __init__(
        self,
        intake: IntakeTurbine,
        air_filter: AirFilter,
        compressor: CompressorTurbine,
        intercooler: Intercooler,
        combustion: CombustionChamber,
        gradient_turbine: GradientTurbine,
        wastegate: WastegateValve,
    ):
        self.intake = intake
        self.filter = air_filter
        self.compressor = compressor
        self.intercooler = intercooler
        self.combustion = combustion
        self.gradient_turbine = gradient_turbine
        self.wastegate = wastegate

    def run_charge_air(self, x: np.ndarray, y: np.ndarray) -> FlowPacket:
        """
        Flows intake batch through intake, filter, compressor, and intercooler.
        Streamlined to avoid redundant intermediate copies.
        """
        raw_packet = self.intake.ingest_raw(x, y)
        clean_packet = self.filter.process(raw_packet)
        compressed_packet = self.compressor.process(clean_packet)
        cooled_packet = self.intercooler.process(compressed_packet)
        return cooled_packet

    def run_combustion(
        self,
        main_packet: FlowPacket,
        injected_packets: List[FlowPacket],
        model: Any,
    ) -> CombustionResult:
        """Ignites fuel mixture inside the combustion chamber."""
        return self.combustion.ignite(
            main_packet=main_packet,
            injected_packets=injected_packets,
            model=model,
        )

    def run_exhaust(
        self,
        model: Any,
        fused_packet: FlowPacket,
        boost_ratio: float,
        combustion_result: Optional[CombustionResult] = None,
    ) -> Tuple[float, float, float, bool]:
        """
        Harvests backprop gradients through Twin-Scroll exhaust runners
        and inspects wastegate relief.
        
        Returns:
            (raw_grad_norm, learning_torque, clipped_norm, was_vented)
        """
        grad_norm, learning_torque = self.gradient_turbine.harvest_gradients(
            model=model,
            fused_packet=fused_packet,
            boost_ratio=boost_ratio,
        )
        exhaust_energy = combustion_result.exhaust_energy if combustion_result is not None else None
        knocking = combustion_result.knocking_detected if combustion_result is not None else False

        clipped_norm, was_vented = self.wastegate.inspect_and_regulate(
            model=model,
            gradient_norm=grad_norm,
            boost_ratio=boost_ratio,
            exhaust_energy=exhaust_energy,
            knocking_detected=knocking,
        )
        return grad_norm, learning_torque, clipped_norm, was_vented


class AuxiliaryInjectionRoundabout:
    """
    Tier 1: Auxiliary Injection Roundabout.
    An independent orbital ring of asynchronous injectors (synthetic perturbations,
    real-world reservoirs, and entropy shock). Connects to Tier 0 via the
    swirl atomization slipway without stalling the core stream.
    """

    def __init__(self, injectors: Optional[List[AsyncDataInjector]] = None):
        self.injectors: List[AsyncDataInjector] = injectors or []
        self.shock_injector: Optional[EntropyShockInjector] = next(
            (inj for inj in self.injectors if isinstance(inj, EntropyShockInjector)), None
        )

    def trigger_nos(self) -> None:
        """Manually forces a high-entropy shock pulse."""
        if self.shock_injector:
            self.shock_injector.trigger_manual_shock()

    def pulse(self, step: int, context_packet: FlowPacket) -> Tuple[List[FlowPacket], float]:
        """
        Polls independent phase clocks of all injectors on this ring.
        Returns active pulses and cumulative injected entropy.
        """
        active_pulses: List[FlowPacket] = []
        total_entropy = 0.0
        for inj in self.injectors:
            packet = inj.pulse(step, context_packet=context_packet)
            if packet is not None:
                active_pulses.append(packet)
                total_entropy += float(packet.temperature * packet.batch_size)
        return active_pulses, total_entropy

    def adapt(self, loss_improved: bool) -> None:
        """Adapts pulse dynamics based on whether combustion loss improved."""
        reward = 1.0 if loss_improved else -0.5
        for inj in self.injectors:
            inj.adapt_dynamics(reward)


class ResonantObservationRoundabout:
    """
    Tier 2: Resonant Observation Roundabout.
    Spans above the data circuits as an ambient harmonic field:
      - Mechanical DriveShaft dynamics (angular acceleration & kinetic inertia)
      - Sequential Turbocharger staging (HP low-inertia + LP compound)
      - Dynamic Variable Valve Timing (Cam advance & volumetric efficiency)
      - Braided DNA Helices Controller (Multi-strand phase resonance & VGT apertures)
    """

    def __init__(
        self,
        shaft: DriveShaft,
        braided_ecu: BraidedDNAController,
        sequential_turbo: Optional[SequentialTurboSystem] = None,
        vvt: Optional[VariableValveTiming] = None,
    ):
        self.shaft = shaft
        self.braided_ecu = braided_ecu
        self.sequential_turbo = sequential_turbo
        self.vvt = vvt

    def modulate_intake_window(
        self,
        boost_psi: float,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Applies dynamic VVT cam timing and micro-batch window slicing."""
        vvt_telemetry = {}
        if self.vvt is not None:
            _, vvt_telemetry = self.vvt.update(
                shaft_rpm=self.shaft.rpm,
                boost_psi=boost_psi,
                resonance_index=self.braided_ecu.resonance_index,
            )
            x_intake, y_intake = self.vvt.slice_batch(x_batch, y_batch)
        else:
            x_intake, y_intake = x_batch, y_batch
        return x_intake, y_intake, vvt_telemetry

    def step_rotational_physics(
        self,
        learning_torque: float,
        compressor_load: float,
        dt: float = 0.08,
    ) -> Dict[str, Any]:
        """Integrates Newton-Euler shaft rotation and sequential turbo staging."""
        self.shaft.step(torque_in=learning_torque, load_torque=compressor_load, dt=dt)
        seq_telemetry = {}
        if self.sequential_turbo is not None:
            _, seq_telemetry = self.sequential_turbo.update(
                learning_torque=learning_torque,
                shaft_rpm=self.shaft.rpm,
                dt=dt,
            )
        return seq_telemetry

    def weave_dna(
        self,
        step: int,
        learning_torque: float,
        boost_ratio: float,
        injected_entropy: float,
        loss: float,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Executes one helical braiding cycle across 4 coupled physical strands.
        Returns braid status dict and calculated pyrometer temperature.
        """
        pyrometer_temp = calculate_pyrometer_temp(loss=loss)
        braid_status = self.braided_ecu.update(
            step=step,
            learning_torque=learning_torque,
            boost_ratio=boost_ratio,
            injected_entropy=injected_entropy,
            pyrometer_temp=pyrometer_temp,
            loss=loss,
        )
        return braid_status, pyrometer_temp


class FluidPipeline:
    """
    Multi-Tier Roundabout Fluid Pipeline: Orchestrator of AcceleratorAI.
    
    Coordinates:
      - Tier 0: Express Core Roundabout (intake, compressor, intercooler, combustion, exhaust)
      - Tier 1: Auxiliary Injection Roundabout (independent phase-clocked injectors)
      - Tier 2: Resonant Observation Roundabout (DriveShaft, Sequential Turbo, VVT, Braided DNA)
    """

    def __init__(
        self,
        core_roundabout: ExpressCoreRoundabout,
        injection_roundabout: AuxiliaryInjectionRoundabout,
        observation_roundabout: ResonantObservationRoundabout,
        telemetry_hub: Optional[TelemetryHub] = None,
        telemetry_interval: int = 1,
    ):
        self.core = core_roundabout
        self.injection = injection_roundabout
        self.observation = observation_roundabout
        self.telemetry_hub = telemetry_hub or TelemetryHub()
        self.telemetry_interval = max(1, telemetry_interval)

        self.current_step: int = 0
        self.previous_loss: float = 1.0

    def flow_step(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        model: Any,
        step_index: int,
        epoch_index: int = 0,
    ) -> CombustionResult:
        """
        Executes a single, non-blocking fluid cycle through all 3 roundabout tiers.
        """
        self.current_step = step_index

        # --- Tier 2 -> Tier 0: VVT Dynamic Intake Window Slicing ---
        x_intake, y_intake, vvt_telemetry = self.observation.modulate_intake_window(
            boost_psi=self.core.compressor.boost_psi,
            x_batch=x_batch,
            y_batch=y_batch,
        )

        # --- Tier 0: Charge Air Flow (Intake -> Filter -> Compressor -> Intercooler) ---
        cooled_packet = self.core.run_charge_air(x_intake, y_intake)

        # --- Tier 1: Auxiliary Injection Ring (Parallel Orbit) ---
        injected_packets, injected_entropy = self.injection.pulse(
            step=self.current_step,
            context_packet=cooled_packet,
        )

        # --- Tier 0: Combustion Ignition (Swirl atomization + forward pass) ---
        combustion_result = self.core.run_combustion(
            main_packet=cooled_packet,
            injected_packets=injected_packets,
            model=model,
        )
        loss = combustion_result.loss

        # Inform shock injector of current loss for plateau tracking
        if self.injection.shock_injector:
            self.injection.shock_injector.record_loss(loss)

        # --- Tier 0: Exhaust & Gradient Turbine (Twin-Scroll Divided Runner) ---
        grad_norm, learning_torque, clipped_norm, was_vented = self.core.run_exhaust(
            model=model,
            fused_packet=combustion_result.fused_packet,
            boost_ratio=self.core.compressor.boost_ratio,
            combustion_result=combustion_result,
        )

        # --- Tier 2: Rotational Physics Integration (Shaft & Sequential Turbo) ---
        compressor_load = self.core.compressor.compute_reaction_load()
        seq_telemetry = self.observation.step_rotational_physics(
            learning_torque=learning_torque,
            compressor_load=compressor_load,
            dt=0.08,
        )

        # --- Tier 2: Braided DNA Helical Control Cycle ---
        braid_status, pyrometer_temp = self.observation.weave_dna(
            step=self.current_step,
            learning_torque=learning_torque,
            boost_ratio=self.core.compressor.boost_ratio,
            injected_entropy=injected_entropy,
            loss=loss,
        )

        # Check for Phase Symmetry Breaking (Prolonged Tension)
        if braid_status.get("should_phase_shock", False):
            self.injection.trigger_nos()

        # Variable Geometry Aperture Modulation
        aperture_signals = braid_status.get("aperture_signals", {})
        for port in self.core.compressor.inlet_manifold.ports:
            target = aperture_signals.get(port.mode, None)
            if target is not None:
                port.adjust_towards(target, speed=0.12)

        # Model Parameter Update
        model.apply_updates(learning_rate=braid_status["learning_rate"])

        # Adapt injection dynamics based on gradient reward
        self.injection.adapt(loss_improved=(loss < self.previous_loss))
        self.previous_loss = loss

        # --- Telemetry Emission (Decoupled sampling support) ---
        if self.current_step % self.telemetry_interval == 0:
            self._emit_telemetry(
                step=self.current_step,
                epoch=epoch_index,
                loss=loss,
                learning_torque=learning_torque,
                compressor_load=compressor_load,
                pyrometer_temp=pyrometer_temp,
                injected_entropy=injected_entropy,
                active_injectors=len(injected_packets),
                combustion_result=combustion_result,
                braid_status=braid_status,
                vvt_telemetry=vvt_telemetry,
                seq_telemetry=seq_telemetry,
                cooled_packet=cooled_packet,
            )

        return combustion_result

    def _emit_telemetry(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_torque: float,
        compressor_load: float,
        pyrometer_temp: float,
        injected_entropy: float,
        active_injectors: int,
        combustion_result: CombustionResult,
        braid_status: Dict[str, Any],
        vvt_telemetry: Dict[str, Any],
        seq_telemetry: Dict[str, Any],
        cooled_packet: FlowPacket,
    ) -> None:
        """Constructs and dispatches engine telemetry."""
        hyper_port = self.core.compressor.inlet_manifold.get_port("hyper")
        cruise_port = self.core.compressor.inlet_manifold.get_port("cruise")
        slowmo_port = self.core.compressor.inlet_manifold.get_port("slowmo")
        curriculum_weights = combustion_result.fused_packet.metadata.get("curriculum_weights", None)
        shaft = self.observation.shaft

        telemetry = EngineTelemetry(
            step=step,
            epoch=epoch,
            rpm=float(shaft.rpm),
            boost_psi=float(self.core.compressor.boost_psi),
            manifold_pressure=float(self.core.compressor.boost_ratio),
            pyrometer_temp_c=float(pyrometer_temp),
            loss=float(loss),
            learning_torque_nm=float(learning_torque),
            compressor_load_nm=float(compressor_load),
            shaft_kinetic_energy_j=float(shaft.kinetic_energy),
            angular_accel_rad_s2=float(shaft.last_angular_accel),
            wastegate_open_pct=float(self.core.wastegate.open_pct),
            air_fuel_ratio=float(combustion_result.air_fuel_ratio),
            injected_entropy=float(injected_entropy),
            active_injectors=active_injectors,
            learning_rate=float(braid_status["learning_rate"]),
            helical_resonance=float(braid_status["resonance_index"]),
            phase_tension=float(braid_status["phase_tension"]),
            winding_number=float(braid_status["winding_number"]),
            homogeneity_pct=float(combustion_result.homogeneity_pct),
            hyper_flow_aperture=float(hyper_port.aperture) if hyper_port else 0.15,
            cruise_flow_aperture=float(cruise_port.aperture) if cruise_port else 0.50,
            slowmo_flow_aperture=float(slowmo_port.aperture) if slowmo_port else 0.85,
            hyper_flow_pct=float(
                (hyper_port.last_routed_count / max(1, cooled_packet.batch_size)) * 100.0
            ) if hyper_port else 0.0,
            curriculum_weight_mean=float(np.mean(curriculum_weights)) if curriculum_weights is not None else 1.0,
            sequential_stage=str(seq_telemetry.get("sequential_stage", "HP_PRIMARY")),
            transition_valve_pct=float(seq_telemetry.get("transition_valve_pct", 0.0)),
            hp_rpm=float(seq_telemetry.get("hp_rpm", 1200.0)),
            lp_rpm=float(seq_telemetry.get("lp_rpm", 600.0)),
            cam_advance_deg=float(vvt_telemetry.get("cam_advance_deg", 0.0)),
            valve_lift=float(vvt_telemetry.get("valve_lift", 0.50)),
            volumetric_efficiency=float(vvt_telemetry.get("volumetric_efficiency", 0.85)),
            twin_scroll_balance=float(self.core.gradient_turbine.twin_scroll.pulse_balance) if self.core.gradient_turbine.twin_scroll else 1.0,
        )
        self.telemetry_hub.emit(telemetry)
