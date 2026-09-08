"""
Unit tests for AcceleratorAI SwirlDispersionValve and Atomization System.
"""

import unittest
import numpy as np

from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.turbines.dispersion_valve import SwirlDispersionValve
from accelerator_ai.turbines.combustion import CombustionChamber
from accelerator_ai.models.neural_core import PureNumPyMLP


class TestDispersionValve(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 32
        self.n_features = 8
        self.x_main = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y_main = np.random.randint(0, 2, size=(self.n_samples,)).astype(np.int32)
        self.main_packet = FlowPacket(x=self.x_main, y=self.y_main, pressure=1.2, temperature=25.0)
        self.model = PureNumPyMLP(layer_sizes=[8, 16, 2])

    def test_valve_initialization(self):
        valve = SwirlDispersionValve(
            atomization_radius=0.15,
            swirl_angle_deg=45.0,
            diffusive_mode=True,
            stratified_swirl=True,
        )
        self.assertEqual(valve.atomization_radius, 0.15)
        self.assertEqual(valve.swirl_angle_deg, 45.0)
        self.assertTrue(valve.diffusive_mode)
        self.assertTrue(valve.stratified_swirl)

    def test_diffusive_atomization_preserves_batch_size(self):
        """Synthetic and chaos packets should diffuse micro-droplets without bloating batch size."""
        valve = SwirlDispersionValve(diffusive_mode=True, stratified_swirl=False)
        
        # Synthetic injection packet
        x_synth = np.random.randn(6, self.n_features).astype(np.float32)
        y_synth = np.random.randint(0, 2, size=(6,)).astype(np.int32)
        synth_packet = FlowPacket(
            x=x_synth,
            y=y_synth,
            source="synthetic_injector",
            temperature=30.0,
            metadata={"injector_type": "synthetic"},
        )

        fused = valve.disperse_and_mix(
            main_packet=self.main_packet,
            injected_packets=[synth_packet],
        )
        homogeneity = valve.last_homogeneity_pct

        # In pure diffusive mode, the batch size stays equal to main packet
        self.assertEqual(fused.batch_size, self.n_samples)
        self.assertFalse(np.array_equal(fused.x, self.x_main))
        # Label space should be continuous soft probabilities
        self.assertEqual(fused.y.shape, (self.n_samples,))
        self.assertTrue(np.all(fused.y >= 0.0) and np.all(fused.y <= 1.0))
        self.assertGreater(homogeneity, 40.0)
        self.assertLessEqual(homogeneity, 100.0)

    def test_stratified_swirl_interleaving(self):
        """Real-world edge samples should be interleaved evenly rather than appended as a lump."""
        valve = SwirlDispersionValve(diffusive_mode=False, stratified_swirl=True)

        x_edge = np.ones((4, self.n_features), dtype=np.float32) * 99.0
        y_edge = np.ones((4,), dtype=np.int32)
        edge_packet = FlowPacket(
            x=x_edge,
            y=y_edge,
            source="real_world_reservoir",
            metadata={"injector_type": "reservoir"},
        )

        fused = valve.disperse_and_mix(
            main_packet=self.main_packet,
            injected_packets=[edge_packet],
        )
        homogeneity = valve.last_homogeneity_pct

        # Batch size expands by 4
        self.assertEqual(fused.batch_size, self.n_samples + 4)
        
        # Check that the edge samples (value 99.0) are NOT all clustered at the end
        is_edge = np.isclose(fused.x[:, 0], 99.0)
        edge_indices = np.where(is_edge)[0]
        self.assertEqual(len(edge_indices), 4)

        # Check spacing: edge samples should be distributed across the batch
        # If clustered at end, indices would be [32, 33, 34, 35]
        self.assertFalse(np.array_equal(edge_indices, np.array([32, 33, 34, 35])))
        # Check minimum gap between injected items
        spacings = np.diff(edge_indices)
        self.assertTrue(np.all(spacings >= 2))

    def test_pure_main_packet_homogeneity(self):
        """No auxiliary injection should return 100% homogeneity."""
        valve = SwirlDispersionValve()
        fused = valve.disperse_and_mix(
            main_packet=self.main_packet,
            injected_packets=None,
        )
        homogeneity = valve.last_homogeneity_pct
        self.assertEqual(homogeneity, 100.0)
        self.assertEqual(fused.batch_size, self.n_samples)

    def test_combustion_chamber_integration(self):
        """CombustionChamber ignite() uses SwirlDispersionValve and returns valid homogeneity."""
        valve = SwirlDispersionValve(atomization_radius=0.1)
        chamber = CombustionChamber(dispersion_valve=valve)

        x_inj = np.random.randn(4, self.n_features).astype(np.float32)
        y_inj = np.random.randint(0, 2, size=(4,)).astype(np.int32)
        inj_packet = FlowPacket(x=x_inj, y=y_inj, source="synthetic_injector", metadata={"injector_type": "synthetic"})

        res = chamber.ignite(
            main_packet=self.main_packet,
            injected_packets=[inj_packet],
            model=self.model,
        )

        self.assertIsNotNone(res.loss)
        self.assertFalse(np.isnan(res.loss))
        self.assertGreater(res.homogeneity_pct, 40.0)
        self.assertLessEqual(res.homogeneity_pct, 100.0)
        self.assertEqual(chamber.ignition_count, 1)


if __name__ == "__main__":
    unittest.main()
