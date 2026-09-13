"""
Unit tests for Distributed Master-ECU Coordinator and multi-GPU synchronization.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from accelerator_ai.ecu.distributed import DistributedECUCoordinator
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.models.neural_core import PureNumPyMLP


class TestDistributedECUCoordinator(unittest.TestCase):

    def test_standalone_coordinator_properties(self):
        """In single-GPU or CPU mode, coordinator must report non-distributed and master."""
        coord = DistributedECUCoordinator(master_rank=0)
        self.assertFalse(coord.is_distributed)
        self.assertTrue(coord.is_master)
        self.assertEqual(coord.rank, 0)
        self.assertEqual(coord.world_size, 1)

        # Standalone broadcast returns exact inputs with zero communication
        gear, lr, wg, shock = coord.broadcast_engine_state(
            vvt_gear=3,
            learning_rate=0.025,
            wastegate_open=True,
            shock_fired=False,
        )
        self.assertEqual(gear, 3)
        self.assertEqual(lr, 0.025)
        self.assertTrue(wg)
        self.assertFalse(shock)

        telemetry = coord.get_telemetry()
        self.assertFalse(telemetry["is_distributed"])
        self.assertEqual(telemetry["rank"], 0)

    def test_engine_integration_with_mock_worker_rank(self):
        """Simulates worker rank 1 receiving Master ECU's broadcast decisions."""
        mock_coord = MagicMock(spec=DistributedECUCoordinator)
        mock_coord.is_distributed = True
        mock_coord.is_master = False
        mock_coord.rank = 1
        mock_coord.world_size = 4
        # Master commands Gear 3 (batch 64) and learning rate 0.035
        mock_coord.broadcast_engine_state.return_value = (3, 0.035, False, False)

        model = PureNumPyMLP(layer_sizes=[4, 8, 2])
        engine = TurboLearningEngine(
            model=model,
            enable_vvt=True,
            distributed_coordinator=mock_coord,
        )

        x = np.random.randn(64, 4).astype(np.float32)
        y = np.random.randint(0, 2, size=(64, 1)).astype(np.float32)

        res = engine.step(x, y)
        self.assertIsNotNone(res)

        # Worker rank must have adopted Master's gear (Gear 3 = 64) and learning rate (0.035)
        self.assertEqual(engine.vvt.current_gear, 3)
        self.assertEqual(engine.vvt.current_batch_size, 64)
        self.assertEqual(engine.braided_ecu.current_learning_rate, 0.035)
        mock_coord.broadcast_engine_state.assert_called_once()

    def test_sync_interval_throttling(self):
        """Verifies that non-emergency states are throttled when sync_interval > 1."""
        coord = DistributedECUCoordinator(master_rank=0, sync_interval=5)
        # Mock distributed status
        coord._is_distributed = True
        coord._rank = 0
        coord._world_size = 2
        coord._last_synced_state = (2, 0.02, False, False)

        with patch("torch.distributed.broadcast") as mock_dist_broadcast:
            # Step 1: 1 % 5 != 0 -> should NOT broadcast, should return cached
            gear, lr, wg, shock = coord.broadcast_engine_state(
                vvt_gear=3, learning_rate=0.03, wastegate_open=False, shock_fired=False, step=1
            )
            mock_dist_broadcast.assert_not_called()
            self.assertEqual(gear, 2)
            self.assertEqual(lr, 0.02)

            # Step 5: 5 % 5 == 0 -> should broadcast
            coord.broadcast_engine_state(
                vvt_gear=3, learning_rate=0.03, wastegate_open=False, shock_fired=False, step=5
            )
            mock_dist_broadcast.assert_called_once()

            # Emergency: wastegate_open=True at step 2 -> should bypass throttling and broadcast immediately
            mock_dist_broadcast.reset_mock()
            coord.broadcast_engine_state(
                vvt_gear=1, learning_rate=0.01, wastegate_open=True, shock_fired=False, step=2
            )
            mock_dist_broadcast.assert_called_once()


if __name__ == "__main__":
    unittest.main()
