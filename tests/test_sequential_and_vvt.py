"""
Unit tests for AcceleratorAI Sequential Turbocharging (HP/LP),
Variable Valve Timing (VVT), and Twin-Scroll Gradient Exhaust Runner.
"""

import unittest
import numpy as np

from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.turbines.sequential_turbo import SequentialTurboSystem, HPTurbo, LPTurbo
from accelerator_ai.turbines.vvt import VariableValveTiming
from accelerator_ai.turbines.gradient_turbine import GradientTurbine, TwinScrollHousing
from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.engine import TurboLearningEngine


class TestSequentialAndVVT(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 32
        self.n_features = 6
        self.x = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y = np.random.randint(0, 2, size=(self.n_samples,)).astype(np.int32)
        self.packet = FlowPacket(x=self.x, y=self.y, pressure=1.0, temperature=25.0)

    def test_hp_turbo_fast_spool_vs_lp(self):
        """Low-inertia HP turbo must spool significantly faster than high-inertia LP turbo."""
        hp = HPTurbo(inertia=0.012, idle_rpm=1200.0)
        lp = LPTurbo(inertia=0.085, idle_rpm=600.0)

        torque = 15.0  # Constant applied torque
        dt = 0.05

        # Step 5 cycles
        for _ in range(5):
            hp.update(torque, dt=dt)
            lp.update(torque, dt=dt)

        hp_rpm_gain = hp.rpm - hp.idle_rpm
        lp_rpm_gain = lp.rpm - lp.idle_rpm

        self.assertGreater(hp_rpm_gain, lp_rpm_gain * 2.5)
        self.assertGreater(hp.boost_ratio, 1.05)

    def test_sequential_turbo_system_stages(self):
        """Test sequential transition valve across HP_PRIMARY, TRANSITION, and LP_COMPOUND."""
        seq = SequentialTurboSystem(crossover_rpm=2200.0, full_transition_rpm=3400.0)

        # 1. Low RPM regime (< 2200): HP_PRIMARY, valve closed
        boost1, tel1 = seq.update(learning_torque=10.0, shaft_rpm=1500.0, dt=0.05)
        self.assertEqual(tel1["sequential_stage"], "HP_PRIMARY")
        self.assertEqual(tel1["transition_valve_pct"], 0.0)

        # 2. Medium RPM regime (2800): TRANSITION, valve partially open (50%)
        boost2, tel2 = seq.update(learning_torque=12.0, shaft_rpm=2800.0, dt=0.05)
        self.assertEqual(tel2["sequential_stage"], "TRANSITION")
        self.assertAlmostEqual(tel2["transition_valve_pct"], 50.0, places=1)

        # 3. High RPM regime (> 3400): LP_COMPOUND, valve fully open (100%)
        boost3, tel3 = seq.update(learning_torque=14.0, shaft_rpm=4000.0, dt=0.05)
        self.assertEqual(tel3["sequential_stage"], "LP_COMPOUND")
        self.assertEqual(tel3["transition_valve_pct"], 100.0)
        self.assertGreater(boost3, boost1)

    def test_vvt_cam_phasing_and_lift(self):
        """Test VVT cam advance angle and valve lift across RPM spectrum."""
        vvt = VariableValveTiming(base_batch_size=32, min_batch_size=16, max_batch_size=64)

        # Low RPM (1000 RPM, 0 PSI) -> Retarded cam, low lift
        b_low, tel_low = vvt.update(shaft_rpm=1000.0, boost_psi=0.0, resonance_index=0.0)
        self.assertLess(tel_low["cam_advance_deg"], 0.0)
        self.assertLessEqual(tel_low["valve_lift"], 0.50)
        self.assertLessEqual(b_low, 32)

        # High RPM (4500 RPM, 18 PSI, constructive resonance H=+0.5) -> Advanced cam, high lift
        b_high, tel_high = vvt.update(shaft_rpm=4500.0, boost_psi=18.0, resonance_index=0.5)
        self.assertGreater(tel_high["cam_advance_deg"], 20.0)
        self.assertGreaterEqual(tel_high["valve_lift"], 0.80)
        self.assertGreater(b_high, b_low)
        self.assertTrue(0.40 <= tel_high["volumetric_efficiency"] <= 1.25)

    def test_vvt_dynamic_batch_slicing(self):
        """Test that VVT properly slices or adjusts batches to match dynamic window."""
        vvt = VariableValveTiming(base_batch_size=32, min_batch_size=16, max_batch_size=48)
        x_in = np.random.randn(64, 4).astype(np.float32)
        y_in = np.random.randint(0, 2, size=64).astype(np.int32)

        # Force current batch size to 20
        vvt.current_batch_size = 20
        x_sliced, y_sliced = vvt.slice_batch(x_in, y_in)
        self.assertEqual(len(x_sliced), 20)
        self.assertEqual(len(y_sliced), 20)
        self.assertEqual(x_sliced.shape[1], 4)

    def test_twin_scroll_gradient_isolation(self):
        """Test that TwinScrollHousing isolates main vs auxiliary injection pulses."""
        twin = TwinScrollHousing()

        # Pure main packet (0 injected samples)
        p_main = FlowPacket(x=self.x, y=self.y, metadata={"injected_samples": 0})
        torque_main, tel_main = twin.divide_and_combine(base_torque=10.0, fused_packet=p_main)
        self.assertAlmostEqual(tel_main["scroll_a_pressure"], 10.0)
        self.assertAlmostEqual(tel_main["scroll_b_pressure"], 0.0)
        self.assertAlmostEqual(tel_main["twin_scroll_balance"], 1.0)

        # Mixed packet with 8 injected samples (8/32 = 25%)
        p_mixed = FlowPacket(x=self.x, y=self.y, metadata={"injected_samples": 8, "shock_fired": True})
        torque_mixed, tel_mixed = twin.divide_and_combine(base_torque=10.0, fused_packet=p_mixed)
        self.assertGreater(tel_mixed["scroll_b_pressure"], 0.0)
        self.assertLess(tel_mixed["twin_scroll_balance"], 1.0)
        self.assertEqual(tel_mixed["injection_pulse_active"], 1.0)

    def test_engine_sequential_and_vvt_closed_loop(self):
        """Full closed-loop verification of TurboLearningEngine with Sequential Turbo and VVT."""
        model = PureNumPyMLP(layer_sizes=[6, 16, 2])
        engine = TurboLearningEngine(
            model=model,
            enable_sequential_turbo=True,
            enable_vvt=True,
            base_learning_rate=0.015,
        )

        res = engine.step(self.x, self.y)
        self.assertIsNotNone(res)
        self.assertGreater(res.loss, 0.0)

        latest = engine.telemetry_hub.latest
        self.assertIsNotNone(latest)
        self.assertIn(latest.sequential_stage, ["HP_PRIMARY", "TRANSITION", "LP_COMPOUND"])
        self.assertTrue(hasattr(latest, "transition_valve_pct"))
        self.assertTrue(hasattr(latest, "hp_rpm"))
        self.assertTrue(hasattr(latest, "lp_rpm"))
        self.assertTrue(hasattr(latest, "cam_advance_deg"))
        self.assertTrue(hasattr(latest, "valve_lift"))
        self.assertTrue(hasattr(latest, "volumetric_efficiency"))
        self.assertTrue(hasattr(latest, "twin_scroll_balance"))

    def test_vvt_discrete_gearbox_and_zero_copy(self):
        """Tests that VVT only selects from discrete gears (16, 32, 64) and uses zero-copy slicing."""
        vvt = VariableValveTiming(base_batch_size=32, min_batch_size=16, max_batch_size=64)
        self.assertEqual(vvt.gears, (16, 32, 64))

        # Gear 1: Low RPM spooling
        b1, tel1 = vvt.update(shaft_rpm=1000.0, boost_psi=0.0)
        self.assertEqual(b1, 16)
        self.assertEqual(tel1["vvt_gear"], 1)

        # Gear 2: Cruise
        b2, tel2 = vvt.update(shaft_rpm=2000.0, boost_psi=5.0)
        self.assertEqual(b2, 32)
        self.assertEqual(tel2["vvt_gear"], 2)

        # Gear 3: Peak boost & VTEC
        b3, tel3 = vvt.update(shaft_rpm=4200.0, boost_psi=20.0, resonance_index=0.5)
        self.assertEqual(b3, 64)
        self.assertEqual(tel3["vvt_gear"], 3)

        # Zero-copy slice test: check slice points to same memory
        x_large = np.arange(640, dtype=np.float32).reshape(64, 10)
        y_large = np.zeros((64,), dtype=np.int32)
        vvt.current_batch_size = 32
        x_slice, y_slice = vvt.slice_batch(x_large, y_large)
        self.assertEqual(len(x_slice), 32)
        self.assertTrue(np.shares_memory(x_large, x_slice))  # Validates zero-copy view!


if __name__ == "__main__":
    unittest.main()

