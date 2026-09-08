"""
Unit tests for AcceleratorAI Variable Geometry Multi-Port Turbine System (VGT).
Tests FlowPort, PortManifold, Venturi velocity dynamics, curriculum weighting,
and closed-loop integration with CompressorTurbine, GradientTurbine, and TurboLearningEngine.
"""

import unittest
import numpy as np

from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.core.flow_port import FlowPort, PortManifold
from accelerator_ai.turbines.compressor import CompressorTurbine
from accelerator_ai.turbines.gradient_turbine import GradientTurbine
from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.ecu.braided_controller import BraidedDNAController
from accelerator_ai.engine import TurboLearningEngine


class TestFlowPorts(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 40
        self.n_features = 8
        self.x = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y = np.random.randint(0, 2, size=(self.n_samples,)).astype(np.int32)
        self.packet = FlowPacket(x=self.x, y=self.y, pressure=1.0, temperature=25.0)

    def test_flow_port_aperture_clamping_and_venturi(self):
        """Test aperture bounds, smooth adjustment, and Venturi velocity calculation."""
        port = FlowPort("test_port", mode="hyper", aperture=0.2, min_aperture=0.05, max_aperture=1.0)
        self.assertAlmostEqual(port.aperture, 0.2)
        self.assertAlmostEqual(port.velocity_factor, 5.0)  # 1 / 0.2 = 5.0

        # Aperture clamping at min and max
        port.aperture = 0.01  # below min
        self.assertAlmostEqual(port.aperture, 0.05)
        self.assertAlmostEqual(port.velocity_factor, 20.0)

        port.aperture = 2.0  # above max
        self.assertAlmostEqual(port.aperture, 1.0)
        self.assertAlmostEqual(port.velocity_factor, 1.0)

        # Smooth adjustment towards target
        port.aperture = 0.5
        port.adjust_towards(target=0.1, speed=0.5)
        self.assertAlmostEqual(port.aperture, 0.3)

    def test_port_admission_mask(self):
        """Test that port admits only samples within its selectivity pressure range."""
        port = FlowPort("hyper", mode="hyper", selectivity_range=(0.7, 1.0))
        pressures = np.array([0.1, 0.5, 0.7, 0.85, 1.0, 0.69])
        mask = port.admits(pressures)
        expected = np.array([False, False, True, True, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_port_manifold_pressure_computation(self):
        """Test L2 feature norm pressure calculation on packets."""
        # Create packet where first sample has tiny norm and last sample has large norm
        x_custom = np.zeros((3, 4), dtype=np.float32)
        x_custom[0] = [0.1, 0.0, 0.0, 0.0]    # norm = 0.1
        x_custom[1] = [1.0, 1.0, 1.0, 1.0]    # norm = 2.0
        x_custom[2] = [4.0, 0.0, 0.0, 0.0]    # norm = 4.0
        p = FlowPacket(x=x_custom, y=np.array([0, 1, 0]))

        pressures = PortManifold.compute_sample_pressure(p)
        self.assertEqual(len(pressures), 3)
        self.assertAlmostEqual(pressures[0], 0.0)
        self.assertAlmostEqual(pressures[2], 1.0)
        self.assertTrue(0.0 <= pressures[1] <= 1.0)

        # Test uniform norm fallback (e.g. constant vectors)
        x_uniform = np.ones((4, 4), dtype=np.float32)
        p_uniform = FlowPacket(x=x_uniform, y=np.zeros(4))
        pressures_uniform = PortManifold.compute_sample_pressure(p_uniform)
        np.testing.assert_array_equal(pressures_uniform, np.full(4, 0.5))

    def test_manifold_route_and_transform(self):
        """Test routing across hyper, cruise, and slow-mo ports and curriculum weights."""
        manifold = PortManifold(side="inlet")
        transformed = manifold.route_and_transform(self.packet)

        self.assertIn("curriculum_weights", transformed.metadata)
        self.assertIn("sample_pressures", transformed.metadata)
        self.assertIn("port_assignments", transformed.metadata)

        weights = transformed.metadata["curriculum_weights"]
        self.assertEqual(len(weights), self.n_samples)
        # Normalized mean should be approx 1.0
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=4)

        port_assignments = transformed.metadata["port_assignments"]
        # Hyper port has index 0, cruise has 1, slowmo has 2
        hyper_indices = np.where(port_assignments == 0)[0]
        slowmo_indices = np.where(port_assignments == 2)[0]

        if len(hyper_indices) > 0 and len(slowmo_indices) > 0:
            # Hyper samples must receive higher curriculum weight than slow-mo samples
            self.assertGreater(weights[hyper_indices[0]], weights[slowmo_indices[0]])

    def test_compressor_with_vgt_manifold(self):
        """Test CompressorTurbine integration with VGT inlet manifold."""
        compressor = CompressorTurbine(base_boost_ratio=1.5, enable_augmentation=True)
        boosted = compressor.process(self.packet)

        self.assertIn("curriculum_weights", boosted.metadata)
        self.assertIn("hyper_flow_aperture", compressor.last_telemetry)
        self.assertIn("cruise_flow_aperture", compressor.last_telemetry)
        self.assertIn("slowmo_flow_aperture", compressor.last_telemetry)
        self.assertIn("hyper_flow_samples", compressor.last_telemetry)

    def test_neural_core_weighted_forward_and_backward(self):
        """Test PureNumPyMLP forward_and_loss and backward with sample curriculum weights."""
        model = PureNumPyMLP(layer_sizes=[8, 16, 2])
        probs, unweighted_loss = model.forward_and_loss(self.x, self.y)
        grad_norm_unweighted = model.backward()

        # Weighted: double weights on first half, zero on second half
        weights = np.zeros(self.n_samples, dtype=np.float64)
        weights[: self.n_samples // 2] = 2.0
        weights_mean = np.mean(weights)
        weights = weights / weights_mean

        probs_w, weighted_loss = model.forward_and_loss(self.x, self.y, sample_weights=weights)
        grad_norm_weighted = model.backward()

        self.assertIsInstance(weighted_loss, float)
        self.assertIsInstance(grad_norm_weighted, float)
        self.assertGreater(grad_norm_weighted, 0.0)

    def test_gradient_turbine_curriculum_torque(self):
        """Test that higher curriculum weights scale the extracted learning torque."""
        gt = GradientTurbine()
        model = PureNumPyMLP(layer_sizes=[8, 16, 2])

        # High curriculum weight packet
        p_high = FlowPacket(x=self.x, y=self.y, metadata={"curriculum_weights": np.full(self.n_samples, 1.8)})
        model.forward_and_loss(self.x, self.y)
        _, torque_high = gt.harvest_gradients(model, p_high, boost_ratio=1.5)

        # Low curriculum weight packet
        p_low = FlowPacket(x=self.x, y=self.y, metadata={"curriculum_weights": np.full(self.n_samples, 0.5)})
        model.forward_and_loss(self.x, self.y)
        _, torque_low = gt.harvest_gradients(model, p_low, boost_ratio=1.5)

        self.assertGreater(torque_high, torque_low)
        self.assertAlmostEqual(gt.last_telemetry["curriculum_torque_factor"], 0.5)

    def test_braided_dna_aperture_signals(self):
        """Test that BraidedDNAController emits adaptive aperture signals."""
        ecu = BraidedDNAController()
        status = ecu.update(
            step=1,
            learning_torque=1.5,
            boost_ratio=1.8,
            injected_entropy=0.5,
            pyrometer_temp=350.0,
            loss=0.45,
        )
        self.assertIn("aperture_signals", status)
        signals = status["aperture_signals"]
        self.assertIn("hyper", signals)
        self.assertIn("cruise", signals)
        self.assertIn("slowmo", signals)
        self.assertTrue(0.05 <= signals["hyper"] <= 0.40)
        self.assertAlmostEqual(signals["cruise"], 0.50)
        self.assertTrue(0.50 <= signals["slowmo"] <= 0.95)

    def test_engine_closed_loop_with_vgt(self):
        """Full TurboLearningEngine step verification with VGT multi-port dynamics."""
        model = PureNumPyMLP(layer_sizes=[8, 16, 2])
        engine = TurboLearningEngine(
            model=model,
            base_learning_rate=0.01,
        )

        res = engine.step(self.x, self.y)
        self.assertIsNotNone(res)
        self.assertGreater(res.loss, 0.0)

        # Check telemetry fields emitted by engine
        hub = engine.telemetry_hub
        latest = hub.latest
        self.assertIsNotNone(latest)
        self.assertTrue(hasattr(latest, "hyper_flow_aperture"))
        self.assertTrue(hasattr(latest, "cruise_flow_aperture"))
        self.assertTrue(hasattr(latest, "slowmo_flow_aperture"))
        self.assertTrue(hasattr(latest, "hyper_flow_pct"))
        self.assertTrue(hasattr(latest, "curriculum_weight_mean"))

        self.assertGreater(latest.hyper_flow_aperture, 0.0)
        self.assertGreater(latest.curriculum_weight_mean, 0.0)


if __name__ == "__main__":
    unittest.main()
