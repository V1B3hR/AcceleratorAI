"""
Unit tests for InputGuard security and validation module.
"""

import unittest
import numpy as np

from accelerator_ai.security.input_guard import (
    InputGuard,
    ValidationError,
    CorruptedTensorError,
)


class TestInputGuard(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.guard = InputGuard(expected_features=8, max_magnitude=100.0)

    def test_valid_2d_batch(self):
        x = np.random.randn(16, 8).astype(np.float32)
        y = np.random.randint(0, 2, size=(16,)).astype(np.int32)
        clean_x, clean_y = self.guard.sanitize(x, y)

        self.assertEqual(clean_x.shape, (16, 8))
        self.assertEqual(clean_y.shape, (16,))
        self.assertEqual(self.guard.total_screened_batches, 1)
        self.assertEqual(self.guard.total_rejected_batches, 0)

    def test_expand_1d_sample(self):
        x = np.random.randn(8)
        clean_x, clean_y = self.guard.sanitize(x, None)
        self.assertEqual(clean_x.shape, (1, 8))

    def test_feature_dimension_mismatch(self):
        x_wrong = np.random.randn(16, 5)  # Expected 8
        with self.assertRaises(ValidationError):
            self.guard.sanitize(x_wrong)
        self.assertEqual(self.guard.total_rejected_batches, 1)

    def test_batch_size_mismatch(self):
        x = np.random.randn(16, 8)
        y_short = np.array([0, 1])  # Only 2 samples
        with self.assertRaises(ValidationError):
            self.guard.sanitize(x, y_short)

    def test_nan_scrubbing_lenient_mode(self):
        x_nan = np.random.randn(10, 8)
        x_nan[0, 0] = np.nan
        x_nan[1, 1] = np.inf

        clean_x, _ = self.guard.sanitize(x_nan)
        self.assertFalse(np.isnan(clean_x).any())
        self.assertFalse(np.isinf(clean_x).any())
        self.assertGreater(self.guard.total_sanitized_samples, 0)

    def test_nan_rejection_strict_mode(self):
        strict_guard = InputGuard(strict_mode=True, allow_nan=False)
        x_nan = np.random.randn(10, 8)
        x_nan[2, 3] = np.nan

        with self.assertRaises(CorruptedTensorError):
            strict_guard.sanitize(x_nan)

    def test_magnitude_clamping(self):
        x_extreme = np.random.randn(10, 8)
        x_extreme[0, 0] = 5000.0  # max_magnitude is 100.0

        clean_x, _ = self.guard.sanitize(x_extreme)
        self.assertLessEqual(np.max(clean_x), 100.0)

    def test_inspect_health(self):
        x = np.random.randn(20, 8).astype(np.float32)
        health = self.guard.inspect_health(x)
        self.assertEqual(health["batch_size"], 20)
        self.assertEqual(health["features"], 8)
        self.assertFalse(health["has_nans"])


if __name__ == "__main__":
    unittest.main()
