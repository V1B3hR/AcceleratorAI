"""
Unit tests for EngineConfig, Exception hierarchy, and enhanced Security validation.
"""

import unittest
import os
import numpy as np
import torch

from accelerator_ai.config import EngineConfig
from accelerator_ai.exceptions import (
    AcceleratorAIError,
    ConfigurationError,
    SecurityValidationError,
    CorruptedTensorError,
)
from accelerator_ai.security import InputGuard
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.models.neural_core import PureNumPyMLP


class TestConfigAndSecurity(unittest.TestCase):

    def test_engine_config_defaults_and_validation(self):
        """Verifies EngineConfig initializes cleanly and catches invalid parameters."""
        cfg = EngineConfig()
        self.assertEqual(cfg.base_learning_rate, 0.015)
        self.assertEqual(cfg.target_boost_psi, 14.7)
        self.assertEqual(cfg.gears, [16, 32, 64])

        # Test invalid learning rate
        with self.assertRaises(ConfigurationError):
            EngineConfig(base_learning_rate=-0.01)

        # Test invalid boost
        with self.assertRaises(ConfigurationError):
            EngineConfig(target_boost_psi=150.0)

        # Test invalid telemetry interval
        with self.assertRaises(ConfigurationError):
            EngineConfig(telemetry_interval=0)

    def test_engine_config_serialization_and_env(self):
        """Verifies dict roundtrip and environment variable overrides."""
        cfg = EngineConfig(base_learning_rate=0.025, target_boost_psi=20.0)
        d = cfg.to_dict()
        self.assertEqual(d["base_learning_rate"], 0.025)

        restored = EngineConfig.from_dict(d)
        self.assertEqual(restored.base_learning_rate, 0.025)
        self.assertEqual(restored.target_boost_psi, 20.0)

        # Env variable override
        os.environ["ACCELERATOR_BASE_LEARNING_RATE"] = "0.035"
        os.environ["ACCELERATOR_TARGET_BOOST_PSI"] = "25.5"
        env_cfg = EngineConfig.from_env()
        self.assertAlmostEqual(env_cfg.base_learning_rate, 0.035)
        self.assertAlmostEqual(env_cfg.target_boost_psi, 25.5)
        del os.environ["ACCELERATOR_BASE_LEARNING_RATE"]
        del os.environ["ACCELERATOR_TARGET_BOOST_PSI"]

    def test_engine_initialization_with_config(self):
        """Verifies TurboLearningEngine accepts and adopts an EngineConfig instance."""
        cfg = EngineConfig(
            base_learning_rate=0.04,
            target_boost_psi=18.0,
            gears=[12, 24, 48],
            fast_physics=True,
        )
        model = PureNumPyMLP(layer_sizes=[4, 8, 2])
        engine = TurboLearningEngine(model=model, config=cfg)

        self.assertEqual(engine.braided_ecu.base_learning_rate, 0.04)
        self.assertTrue(engine.fast_physics)
        self.assertEqual(list(engine.vvt.gears), [12, 24, 48])

    def test_input_guard_batch_and_memory_limits(self):
        """Verifies InputGuard enforces max_batch_size and memory safety limits."""
        guard = InputGuard(max_batch_size=100, max_tensor_bytes=1000)

        # Exceed batch size
        huge_batch = np.zeros((150, 4), dtype=np.float32)
        with self.assertRaises(SecurityValidationError):
            guard.sanitize(huge_batch)

        # Exceed memory bytes (1000 float32 elements = 4000 bytes > 1000 limit)
        large_memory = np.zeros((20, 50), dtype=np.float32)
        with self.assertRaises(SecurityValidationError):
            guard.sanitize(large_memory)

        # PyTorch path limit test
        torch_huge = torch.zeros((120, 2), dtype=torch.float32)
        with self.assertRaises(SecurityValidationError):
            guard.sanitize(torch_huge)

    def test_input_guard_device_mismatch(self):
        """Verifies InputGuard catches device mismatch between X and Y."""
        guard = InputGuard()
        x_cpu = torch.randn(10, 4)
        if torch.cuda.is_available():
            y_cuda = torch.randint(0, 2, (10, 1), device="cuda:0")
            with self.assertRaises(SecurityValidationError):
                guard.sanitize(x_cpu, y_cuda)

    def test_exception_inheritance(self):
        """Verifies all domain exceptions derive from AcceleratorAIError."""
        self.assertTrue(issubclass(ConfigurationError, AcceleratorAIError))
        self.assertTrue(issubclass(SecurityValidationError, AcceleratorAIError))
        self.assertTrue(issubclass(CorruptedTensorError, AcceleratorAIError))


if __name__ == "__main__":
    unittest.main()
