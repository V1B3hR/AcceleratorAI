"""
Unit tests for AcceleratorAI FluidPipeline & Multi-Tier Roundabout Manifold (Vibe Flow).
"""

import unittest
import numpy as np

from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.core.shaft import DriveShaft
from accelerator_ai.core.pipeline import (
    FluidPipeline,
    PipelineStage,
    PipelineChain,
    ExpressCoreRoundabout,
    AuxiliaryInjectionRoundabout,
    ResonantObservationRoundabout,
)
from accelerator_ai.turbines.intake import IntakeTurbine
from accelerator_ai.turbines.filter import AirFilter
from accelerator_ai.turbines.compressor import CompressorTurbine
from accelerator_ai.turbines.intercooler import Intercooler
from accelerator_ai.turbines.combustion import CombustionChamber
from accelerator_ai.turbines.gradient_turbine import GradientTurbine
from accelerator_ai.turbines.wastegate import WastegateValve
from accelerator_ai.turbines.sequential_turbo import SequentialTurboSystem
from accelerator_ai.turbines.vvt import VariableValveTiming
from accelerator_ai.injectors.synthetic import SyntheticInjector
from accelerator_ai.injectors.realworld import RealWorldReservoirInjector
from accelerator_ai.injectors.shock import EntropyShockInjector
from accelerator_ai.ecu.braided_controller import BraidedDNAController
from accelerator_ai.ecu.telemetry import TelemetryHub
from accelerator_ai.models.neural_core import PureNumPyMLP


class TestFluidPipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.x = np.random.randn(32, 8).astype(np.float32)
        self.y = np.random.randint(0, 2, size=(32,)).astype(np.int32)
        self.model = PureNumPyMLP(layer_sizes=[8, 16, 2])

        # Instantiate components for roundabouts
        self.shaft = DriveShaft(inertia=0.08, idle_rpm=800.0)
        self.intake = IntakeTurbine()
        self.filter = AirFilter()
        self.compressor = CompressorTurbine(shaft=self.shaft)
        self.intercooler = Intercooler()
        self.combustion = CombustionChamber()
        self.gradient_turbine = GradientTurbine(shaft=self.shaft, enable_twin_scroll=True)
        self.wastegate = WastegateValve()

        self.compressor.attach_shaft(self.shaft)
        self.gradient_turbine.attach_shaft(self.shaft)

        self.seq_turbo = SequentialTurboSystem()
        self.vvt = VariableValveTiming(base_batch_size=32)
        self.braided_ecu = BraidedDNAController(base_learning_rate=0.015)
        self.telemetry_hub = TelemetryHub()

        self.injectors = [
            SyntheticInjector(batch_size=4),
            RealWorldReservoirInjector(batch_size=4),
            EntropyShockInjector(batch_size=4, threshold=0.85),
        ]

        # Tier 0, 1, 2 Roundabouts
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
            sequential_turbo=self.seq_turbo,
            vvt=self.vvt,
        )

        self.pipeline = FluidPipeline(
            core_roundabout=self.core_roundabout,
            injection_roundabout=self.injection_roundabout,
            observation_roundabout=self.observation_roundabout,
            telemetry_hub=self.telemetry_hub,
            telemetry_interval=1,
        )

    def test_pipeline_stage_rshift_composition(self):
        """Pipeline stages must compose cleanly using the >> operator."""
        s1 = PipelineStage("Stage1", lambda p: FlowPacket(p.x * 2.0, p.y, pressure=p.pressure * 1.5))
        s2 = PipelineStage("Stage2", lambda p: FlowPacket(p.x + 1.0, p.y, pressure=p.pressure + 0.5))
        s3 = PipelineStage("Stage3", lambda p: FlowPacket(p.x / 2.0, p.y, pressure=p.pressure))

        chain = s1 >> s2 >> s3
        self.assertIsInstance(chain, PipelineChain)
        self.assertEqual(len(chain.stages), 3)

        init_packet = FlowPacket(np.array([[2.0, 4.0]]), np.array([1]), pressure=1.0)
        out_packet = chain.process(init_packet)

        # ((2 * 2) + 1) / 2 = 2.5; ((4 * 2) + 1) / 2 = 4.5
        np.testing.assert_allclose(out_packet.x, np.array([[2.5, 4.5]]))
        self.assertAlmostEqual(out_packet.pressure, 2.0)

    def test_express_core_roundabout_flow(self):
        """Tier 0 must flow charge air, ignite combustion, and extract exhaust gradients."""
        cooled = self.core_roundabout.run_charge_air(self.x, self.y)
        self.assertIsInstance(cooled, FlowPacket)
        self.assertEqual(cooled.batch_size, 32)
        self.assertGreater(cooled.pressure, 0.5)

        comb_res = self.core_roundabout.run_combustion(cooled, [], self.model)
        self.assertGreater(comb_res.loss, 0.0)
        self.assertEqual(comb_res.homogeneity_pct, 100.0)

        grad_norm, torque, clipped_norm, was_vented = self.core_roundabout.run_exhaust(
            self.model, comb_res.fused_packet, boost_ratio=1.2
        )
        self.assertGreater(grad_norm, 0.0)
        self.assertGreater(torque, 0.0)
        self.assertFalse(was_vented)

    def test_auxiliary_injection_roundabout(self):
        """Tier 1 must pulse asynchronous injectors and support manual NOS trigger."""
        cooled = self.core_roundabout.run_charge_air(self.x, self.y)
        pulses, entropy = self.injection_roundabout.pulse(step=1, context_packet=cooled)
        self.assertIsInstance(pulses, list)

        # Trigger manual NOS shock
        self.injection_roundabout.trigger_nos()
        self.assertTrue(self.injection_roundabout.shock_injector.force_shock)

        pulses_nos, entropy_nos = self.injection_roundabout.pulse(step=2, context_packet=cooled)
        # Should include shock injector packet
        shock_packets = [p for p in pulses_nos if p.source == "shock_injector"]
        self.assertEqual(len(shock_packets), 1)

    def test_fluid_pipeline_closed_loop_flow_step(self):
        """FluidPipeline must execute complete multi-tier cycle with valid telemetry and state update."""
        received_telemetry = []
        self.telemetry_hub.add_listener(lambda t: received_telemetry.append(t))

        res = self.pipeline.flow_step(
            x_batch=self.x,
            y_batch=self.y,
            model=self.model,
            step_index=1,
            epoch_index=0,
        )
        self.assertGreater(res.loss, 0.0)
        self.assertEqual(len(received_telemetry), 1)
        tel = received_telemetry[0]
        self.assertEqual(tel.step, 1)
        self.assertGreater(tel.rpm, 0.0)
        self.assertGreater(tel.learning_torque_nm, 0.0)
        self.assertIn(tel.sequential_stage, ["HP_PRIMARY", "TRANSITION", "LP_COMPOUND"])

    def test_telemetry_sampling_interval(self):
        """Sampling interval > 1 must throttle telemetry emission to reduce overhead."""
        sampled_hub = TelemetryHub()
        received = []
        sampled_hub.add_listener(lambda t: received.append(t))

        sampled_pipeline = FluidPipeline(
            core_roundabout=self.core_roundabout,
            injection_roundabout=self.injection_roundabout,
            observation_roundabout=self.observation_roundabout,
            telemetry_hub=sampled_hub,
            telemetry_interval=5,
        )

        # Run 10 steps
        for step in range(1, 11):
            sampled_pipeline.flow_step(self.x, self.y, self.model, step_index=step)

        # Steps 5 and 10 should be emitted
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0].step, 5)
        self.assertEqual(received[1].step, 10)

    def test_flow_packet_pytorch_interop(self):
        """FlowPacket must support backend-agnostic conversion and merge."""
        try:
            import torch
            p1 = FlowPacket(x=np.random.randn(8, 4).astype(np.float32), y=np.random.randint(0, 2, 8))
            p2 = FlowPacket(x=np.random.randn(8, 4).astype(np.float32), y=np.random.randint(0, 2, 8))

            # Convert to PyTorch
            p1_torch = p1.to_torch()
            self.assertTrue(p1_torch.is_torch)
            self.assertIsInstance(p1_torch.x, torch.Tensor)

            # Merge mixed or pure torch packets
            merged = FlowPacket.merge([p1_torch, p2])
            self.assertEqual(merged.batch_size, 16)
            self.assertTrue(merged.is_torch)

            # Convert back to NumPy
            p_numpy = merged.to_numpy()
            self.assertFalse(p_numpy.is_torch)
            self.assertIsInstance(p_numpy.x, np.ndarray)
        except ImportError:
            # Skip gracefully if PyTorch is not in the environment
            pass


if __name__ == "__main__":
    unittest.main()
