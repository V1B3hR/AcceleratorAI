"""
Unit tests for AcceleratorAI Turbines & FlowPacket.
"""

import unittest
import numpy as np

from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.turbines.intake import IntakeTurbine
from accelerator_ai.turbines.filter import AirFilter
from accelerator_ai.turbines.compressor import CompressorTurbine
from accelerator_ai.turbines.intercooler import Intercooler
from accelerator_ai.turbines.combustion import CombustionChamber
from accelerator_ai.turbines.gradient_turbine import GradientTurbine
from accelerator_ai.turbines.wastegate import WastegateValve
from accelerator_ai.models.neural_core import PureNumPyMLP


class TestTurbines(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.x = np.random.randn(32, 8).astype(np.float32)
        self.y = np.random.randint(0, 2, size=(32,)).astype(np.int32)
        self.packet = FlowPacket(x=self.x, y=self.y, pressure=1.0)
        self.model = PureNumPyMLP(layer_sizes=[8, 16, 2])

    def test_flow_packet_merge_and_split(self):
        p1, p2 = self.packet.split(0.5)
        self.assertEqual(p1.batch_size, 16)
        self.assertEqual(p2.batch_size, 16)

        merged = FlowPacket.merge([p1, p2])
        self.assertEqual(merged.batch_size, 32)
        self.assertAlmostEqual(merged.pressure, 1.0)

    def test_intake_turbine(self):
        intake = IntakeTurbine(throttle_pct=80.0)
        out_packet = intake.process(self.packet)
        self.assertAlmostEqual(out_packet.pressure, 0.8)
        self.assertGreater(intake.rpm, 0.0)

    def test_air_filter_purges_nans_and_clamps_outliers(self):
        corrupted_x = self.x.copy()
        corrupted_x[0, 0] = np.nan
        corrupted_x[1, 1] = 1e6  # extreme outlier

        corrupted_packet = FlowPacket(x=corrupted_x, y=self.y)
        air_filter = AirFilter(outlier_std_threshold=3.0)
        clean_packet = air_filter.process(corrupted_packet)

        self.assertFalse(np.isnan(clean_packet.x).any())
        self.assertLess(np.max(clean_packet.x), 1e5)
        self.assertGreater(clean_packet.metadata["scrubbed_particles"], 0)

    def test_compressor_boosts_pressure(self):
        compressor = CompressorTurbine(base_boost_ratio=1.8)
        self.assertAlmostEqual(compressor.boost_psi, (1.8 - 1.0) * 14.7, places=2)

        boosted_packet = compressor.process(self.packet)
        self.assertAlmostEqual(boosted_packet.pressure, 1.8, places=2)
        self.assertGreater(compressor.rpm, 0.0)

    def test_intercooler_normalizes_and_drops_temperature(self):
        initial_temp = self.packet.temperature
        intercooler = Intercooler(cooling_efficiency=0.8)
        cooled_packet = intercooler.process(self.packet)
        self.assertLess(cooled_packet.temperature, initial_temp)
        self.assertEqual(cooled_packet.x.shape, self.x.shape)

    def test_combustion_chamber_ignite(self):
        chamber = CombustionChamber()
        res = chamber.ignite(self.packet, injected_packets=None, model=self.model)
        self.assertGreater(res.loss, 0.0)
        self.assertEqual(res.predictions.shape, (32, 2))
        self.assertGreater(res.exhaust_energy, 0.0)

    def test_gradient_turbine_and_wastegate(self):
        # Forward pass on model
        self.model.forward_and_loss(self.x, self.y)
        grad_turbine = GradientTurbine()
        grad_norm, torque = grad_turbine.harvest_gradients(
            model=self.model, fused_packet=self.packet, boost_ratio=1.5
        )
        self.assertGreater(grad_norm, 0.0)
        self.assertGreater(torque, 0.0)

        # Wastegate regulation
        wastegate = WastegateValve(max_gradient_norm=0.01)  # tiny threshold to force relief
        clipped_norm, was_vented = wastegate.inspect_and_regulate(
            model=self.model, gradient_norm=grad_norm, boost_ratio=1.5
        )
        self.assertTrue(was_vented)
        self.assertLessEqual(clipped_norm, 0.01 + 1e-6)
        self.assertGreater(wastegate.open_pct, 0.0)

    def test_air_filter_magnetic_and_ultrasonic_stages(self):
        # 1. Magnetic Stage (Heavy Metal / Multivariate screening)
        filter_mag = AirFilter(
            outlier_std_threshold=10.0,      # High MAD threshold so 1D mesh lets it pass
            enable_magnetic_stage=True,
            magnetic_threshold_sigma=2.0,     # Tight magnetic trap
            enable_ultrasonic_stage=True,
            ultrasonic_clean_interval=5,
        )
        heavy_x = self.x.copy()
        # Sample 0 has all dimensions elevated to 2.5 sigma: individually ok (<10.0),
        # but joint multivariate distance exceeds 2.0 sigma threshold
        heavy_x[0] = np.mean(heavy_x, axis=0) + 2.5 * np.std(heavy_x, axis=0)
        # Ultrasonic duplication test: make sample 5 identical to sample 1
        heavy_x[5] = heavy_x[1].copy()

        packet = FlowPacket(x=heavy_x, y=self.y)
        clean = filter_mag.process(packet)

        # Verify magnetic trap caught the multi-feature anomaly
        self.assertGreater(clean.metadata["magnetic_trapped"], 0)
        # Verify ultrasonic sonication dispersed the identical duplicate
        self.assertGreater(clean.metadata["sonicated_clusters"], 0)
        self.assertFalse(np.allclose(clean.x[5], clean.x[1]))

        # Piezo Self-Cleaning Pulse test
        filter_mag.clog_level = 0.85
        # Advance packet count to trigger clean interval
        filter_mag.total_processed_packets = 4
        clean2 = filter_mag.process(packet)
        self.assertTrue(clean2.metadata["piezo_cleaned"])
        self.assertLess(filter_mag.clog_level, 0.15)
        self.assertGreaterEqual(filter_mag.efficiency, 0.95)

    def test_combustion_knocking_and_wastegate_relief(self):
        chamber = CombustionChamber(knock_energy_threshold=0.1)  # sensitive threshold
        packet_high_p = FlowPacket(x=self.x, y=self.y, pressure=2.5)
        res = chamber.ignite(packet_high_p, injected_packets=None, model=self.model)

        self.assertGreater(res.information_density, 1.0)
        self.assertTrue(res.knocking_detected)
        self.assertGreater(res.exhaust_energy, 0.1)

        # Wastegate knock relief
        wastegate = WastegateValve(max_gradient_norm=10.0, max_exhaust_energy=0.1)
        clipped_norm, was_vented = wastegate.inspect_and_regulate(
            model=self.model,
            gradient_norm=1.5,
            boost_ratio=2.5,
            exhaust_energy=res.exhaust_energy,
            knocking_detected=res.knocking_detected,
        )
        self.assertTrue(was_vented)
        self.assertGreaterEqual(wastegate.open_pct, 70.0)
        self.assertEqual(wastegate.total_knock_mitigations, 1)
        self.assertTrue(wastegate.last_telemetry["knock_mitigated"])


if __name__ == "__main__":
    unittest.main()

