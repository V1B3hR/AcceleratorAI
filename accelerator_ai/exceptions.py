"""
AcceleratorAI Exception Hierarchy: Standardized domain-specific exceptions.
Ensures clean error handling, logging, and graceful degradation across the engine.
"""


class AcceleratorAIError(Exception):
    """Base exception for all AcceleratorAI framework errors."""
    pass


class ConfigurationError(AcceleratorAIError, ValueError):
    """Raised when engine or turbine configuration parameters are invalid."""
    pass


class SecurityValidationError(AcceleratorAIError, ValueError):
    """Raised when incoming tensors violate structural, dimension, or memory limits."""
    pass


# Backward compatibility alias for InputGuard
ValidationError = SecurityValidationError


class CorruptedTensorError(AcceleratorAIError, ValueError):
    """Raised when incoming tensors contain irrecoverable numerical corruptions (NaN/Inf)."""
    pass


class FluidDynamicsError(AcceleratorAIError, RuntimeError):
    """Raised when fluid pipeline, compression, or turbine flow encounters physical stalls."""
    pass


class HardwareSynchronizationError(AcceleratorAIError, RuntimeError):
    """Raised when distributed GPU communication, NCCL broadcast, or CUDA streams fail/timeout."""
    pass
