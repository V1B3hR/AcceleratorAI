"""
Engine Configuration System: Structured, validated configuration schemas
for TurboLearningEngine with YAML, JSON, dictionary, and environment variable support.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import os
import json

from accelerator_ai.exceptions import ConfigurationError


@dataclass
class EngineConfig:
    """
    Centralized, strongly-typed configuration for TurboLearningEngine.
    Enforces physical parameter bounds and supports multi-environment serialization.
    """
    base_learning_rate: float = 0.015
    target_boost_psi: float = 14.7
    shaft_inertia: float = 0.08
    enable_default_injectors: bool = True
    enable_sequential_turbo: bool = True
    enable_vvt: bool = True
    telemetry_interval: int = 1
    fault_tolerance_mode: bool = True
    fast_physics: bool = False
    max_batch_size: int = 16384
    gears: List[int] = field(default_factory=lambda: [16, 32, 64])

    # High-Performance Execution & Zero-Sync Pipeline Options
    enable_cuda_graph: bool = False
    adaptive_turbo: bool = True
    adaptive_check_interval: int = 10
    enable_amp: bool = False
    amp_dtype: str = "bfloat16"
    enable_telemetry: bool = True
    async_telemetry: bool = False
    auto_detect_gears: bool = False
    filter_cache_window: int = 20
    hierarchical_turbo: bool = True

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Enforces operational physical bounds on configuration parameters."""
        if self.base_learning_rate <= 0.0:
            raise ConfigurationError(
                f"base_learning_rate must be strictly positive (> 0.0), got {self.base_learning_rate}."
            )
        if self.target_boost_psi < 0.0 or self.target_boost_psi > 60.0:
            raise ConfigurationError(
                f"target_boost_psi must be in range [0.0, 60.0] PSI, got {self.target_boost_psi}."
            )
        if self.shaft_inertia <= 0.0:
            raise ConfigurationError(
                f"shaft_inertia must be strictly positive (> 0.0), got {self.shaft_inertia}."
            )
        if self.telemetry_interval < 1:
            raise ConfigurationError(
                f"telemetry_interval must be >= 1, got {self.telemetry_interval}."
            )
        if not self.gears or any(g <= 0 for g in self.gears):
            raise ConfigurationError(
                f"gears must be a list of positive integers, got {self.gears}."
            )
        if self.adaptive_check_interval < 1:
            raise ConfigurationError(
                f"adaptive_check_interval must be >= 1, got {self.adaptive_check_interval}."
            )
        if self.filter_cache_window < 1:
            raise ConfigurationError(
                f"filter_cache_window must be >= 1, got {self.filter_cache_window}."
            )
        valid_amp = ("bfloat16", "float16", "bf16", "fp16")
        if self.amp_dtype.lower() not in valid_amp:
            raise ConfigurationError(
                f"amp_dtype must be one of {valid_amp}, got '{self.amp_dtype}'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes configuration to Python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineConfig":
        """Instantiates EngineConfig from a dictionary with parameter filtering."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "EngineConfig":
        """Loads configuration from a YAML file (requires PyYAML if available, falls back to JSON-compatible YAML)."""
        if not os.path.exists(path):
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except ImportError:
            # Fallback to JSON parser if yaml is not installed
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        if not isinstance(data, dict):
            raise ConfigurationError(f"Invalid configuration file content (expected mapping): {path}")
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        """Dumps configuration to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            import yaml
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except ImportError:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_env(cls, prefix: str = "ACCELERATOR_") -> "EngineConfig":
        """
        Loads configuration overrides from environment variables.
        Example: ACCELERATOR_BASE_LEARNING_RATE=0.02
        """
        config = cls()
        for field_name in cls.__dataclass_fields__:
            env_var = f"{prefix}{field_name.upper()}"
            if env_var in os.environ:
                raw_val = os.environ[env_var]
                current_val = getattr(config, field_name)
                target_type = type(current_val)

                if target_type == bool:
                    parsed_val = raw_val.lower() in ("1", "true", "yes")
                elif target_type == list:
                    parsed_val = [int(x.strip()) for x in raw_val.split(",")]
                else:
                    parsed_val = target_type(raw_val)

                setattr(config, field_name, parsed_val)
        config.validate()
        return config
