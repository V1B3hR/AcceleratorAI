"""
Security & Input Guard: Validates, sanitizes, and screens incoming tensors
before they enter the turbine medium, protecting neural weights from corruptions,
adversarial NaN injections, and shape mismatches.
"""

from typing import Tuple, Optional, Any, Dict
import numpy as np


class ValidationError(ValueError):
    """Raised when input tensors fail critical structural validation."""
    pass


class CorruptedTensorError(ValueError):
    """Raised when incoming data contains irrecoverable numerical corruptions."""
    pass


class InputGuard:
    """
    Enterprise-grade Input Guard & Tensor Sanitizer.
    Enforces strict typing, dimension alignment, finite bounds, and numerical health.
    """

    def __init__(
        self,
        expected_features: Optional[int] = None,
        max_magnitude: float = 1e6,
        strict_mode: bool = False,
        allow_nan: bool = False,
    ):
        self.expected_features = expected_features
        self.max_magnitude = max_magnitude
        self.strict_mode = strict_mode
        self.allow_nan = allow_nan
        self.total_screened_batches: int = 0
        self.total_rejected_batches: int = 0
        self.total_sanitized_samples: int = 0

    def sanitize(
        self,
        x: Any,
        y: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validates and sanitizes x and y inputs into clean, standardized NumPy arrays.
        
        Args:
            x: Feature matrix (array-like, list, or PyTorch tensor).
            y: Optional label vector/matrix.
            
        Returns:
            Tuple of (sanitized_x, sanitized_y).
            
        Raises:
            ValidationError: In strict mode when structural constraints are violated.
        """
        self.total_screened_batches += 1

        # 1. Convert to NumPy ndarray safely
        x_arr = self._to_numpy(x)
        y_arr = self._to_numpy(y) if y is not None else None

        # 2. Dimensionality Checks
        if x_arr.ndim == 1:
            # Expand 1D single sample into (1, d) batch
            x_arr = x_arr.reshape(1, -1)
        elif x_arr.ndim != 2:
            self.total_rejected_batches += 1
            raise ValidationError(
                f"Input X must be a 2D matrix (batch_size, num_features). Got shape {x_arr.shape}."
            )

        batch_size, num_features = x_arr.shape
        if batch_size == 0:
            self.total_rejected_batches += 1
            raise ValidationError("Input X contains 0 samples (empty batch).")

        if self.expected_features is not None and num_features != self.expected_features:
            self.total_rejected_batches += 1
            raise ValidationError(
                f"Feature dimension mismatch: expected {self.expected_features} features, got {num_features}."
            )

        if y_arr is not None:
            if len(y_arr) != batch_size:
                self.total_rejected_batches += 1
                raise ValidationError(
                    f"Batch size mismatch: X has {batch_size} samples, but Y has {len(y_arr)} samples."
                )

        # 3. Numerical Health Screening (NaNs / Infs)
        has_nan_x = np.isnan(x_arr).any() or np.isinf(x_arr).any()
        if has_nan_x:
            if self.strict_mode and not self.allow_nan:
                self.total_rejected_batches += 1
                raise CorruptedTensorError("Input X contains NaN or infinite values under strict mode.")
            # Sanitize: replace NaNs with median/zero and clamp infinities
            x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=self.max_magnitude, neginf=-self.max_magnitude)
            self.total_sanitized_samples += batch_size

        if y_arr is not None:
            has_nan_y = np.isnan(y_arr).any() or np.isinf(y_arr).any()
            if has_nan_y:
                if self.strict_mode and not self.allow_nan:
                    self.total_rejected_batches += 1
                    raise CorruptedTensorError("Labels Y contain NaN or infinite values under strict mode.")
                y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=1.0, neginf=0.0)

        # 4. Extreme Magnitude Clamping
        extreme_mask = np.abs(x_arr) > self.max_magnitude
        if extreme_mask.any():
            if self.strict_mode:
                self.total_rejected_batches += 1
                raise ValidationError(
                    f"Input X contains values exceeding maximum magnitude threshold ({self.max_magnitude})."
                )
            x_arr = np.clip(x_arr, -self.max_magnitude, self.max_magnitude)
            self.total_sanitized_samples += int(np.sum(extreme_mask))

        # 5. Type normalization (Standardize to float32)
        if not np.issubdtype(x_arr.dtype, np.floating):
            x_arr = x_arr.astype(np.float32)

        return x_arr, y_arr

    def inspect_health(self, x: np.ndarray) -> Dict[str, Any]:
        """Returns diagnostic health metrics for a feature tensor."""
        return {
            "batch_size": x.shape[0] if x.ndim > 0 else 0,
            "features": x.shape[1] if x.ndim > 1 else 0,
            "has_nans": bool(np.isnan(x).any()),
            "has_infs": bool(np.isinf(x).any()),
            "mean_magnitude": float(np.mean(np.abs(x))),
            "max_magnitude": float(np.max(np.abs(x))),
            "total_screened": self.total_screened_batches,
            "total_rejected": self.total_rejected_batches,
            "total_sanitized": self.total_sanitized_samples,
        }

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray:
        """Converts arbitrary tensor or sequence into NumPy array with zero copies when possible."""
        if isinstance(data, np.ndarray):
            return data
        # Handle PyTorch tensors without mandatory torch import
        if hasattr(data, "detach") and hasattr(data, "cpu") and hasattr(data, "numpy"):
            return data.detach().cpu().numpy()
        return np.asarray(data)
