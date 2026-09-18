"""
AcceleratorAI: The VIBE Turbine AI Learning Accelerator Framework.
"""

from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.shaft import DriveShaft
from accelerator_ai.core.metrics import EngineTelemetry
from accelerator_ai.turbines.intake import IntakeTurbine
from accelerator_ai.turbines.filter import AirFilter
from accelerator_ai.turbines.compressor import CompressorTurbine
from accelerator_ai.turbines.intercooler import Intercooler
from accelerator_ai.turbines.combustion import CombustionChamber
from accelerator_ai.turbines.gradient_turbine import GradientTurbine
from accelerator_ai.turbines.wastegate import WastegateValve
from accelerator_ai.turbines.dispersion_valve import SwirlDispersionValve
from accelerator_ai.turbines.sequential_turbo import SequentialTurboSystem, HPTurbo, LPTurbo
from accelerator_ai.turbines.vvt import VariableValveTiming
from accelerator_ai.turbines.gradient_turbine import TwinScrollHousing
from accelerator_ai.core.flow_port import FlowPort, PortManifold
from accelerator_ai.injectors.base_injector import AsyncDataInjector
from accelerator_ai.injectors.synthetic import SyntheticInjector
from accelerator_ai.injectors.realworld import RealWorldReservoirInjector
from accelerator_ai.injectors.shock import EntropyShockInjector
from accelerator_ai.ecu.controller import BoostController
from accelerator_ai.ecu.braided_controller import BraidedDNAController
from accelerator_ai.ecu.distributed import DistributedECUCoordinator
from accelerator_ai.ecu.telemetry import TelemetryHub
from accelerator_ai.core.pipeline import (
    FluidPipeline,
    PipelineStage,
    PipelineChain,
    ExpressCoreRoundabout,
    AuxiliaryInjectionRoundabout,
    ResonantObservationRoundabout,
)
from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.models.torch_adapter import PyTorchTurbineWrapper
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.config import EngineConfig
from accelerator_ai.exceptions import (
    AcceleratorAIError,
    ConfigurationError,
    SecurityValidationError,
    ValidationError,
    CorruptedTensorError,
    FluidDynamicsError,
    HardwareSynchronizationError,
)
from accelerator_ai.security import InputGuard
from accelerator_ai.benchmarks.nanogpt import NanoGPT, GPTConfig
from accelerator_ai.wrapper import wrap
from accelerator_ai.integrations import AcceleratorAICallback, AcceleratorAILightningCallback
from accelerator_ai.core.engine_state import EngineState
from accelerator_ai.core.circuit_breaker import ModuleCircuitBreaker, CircuitState
from accelerator_ai.ecu.kalman import KalmanLossGovernor
from accelerator_ai.security.vram_guard import VRAMPressureGuard, PressureLevel

__version__ = "0.5.0"

__all__ = [
    "FlowPacket",
    "TurbineModule",
    "DriveShaft",
    "EngineTelemetry",
    "FluidPipeline",
    "PipelineStage",
    "PipelineChain",
    "ExpressCoreRoundabout",
    "AuxiliaryInjectionRoundabout",
    "ResonantObservationRoundabout",
    "IntakeTurbine",
    "AirFilter",
    "CompressorTurbine",
    "Intercooler",
    "CombustionChamber",
    "GradientTurbine",
    "TwinScrollHousing",
    "WastegateValve",
    "SwirlDispersionValve",
    "SequentialTurboSystem",
    "HPTurbo",
    "LPTurbo",
    "VariableValveTiming",
    "FlowPort",
    "PortManifold",
    "AsyncDataInjector",
    "SyntheticInjector",
    "RealWorldReservoirInjector",
    "EntropyShockInjector",
    "BoostController",
    "BraidedDNAController",
    "DistributedECUCoordinator",
    "TelemetryHub",
    "PureNumPyMLP",
    "PyTorchTurbineWrapper",
    "TurboLearningEngine",
    "EngineConfig",
    "InputGuard",
    "AcceleratorAIError",
    "ConfigurationError",
    "SecurityValidationError",
    "ValidationError",
    "CorruptedTensorError",
    "FluidDynamicsError",
    "HardwareSynchronizationError",
    "NanoGPT",
    "GPTConfig",
    "wrap",
    "AcceleratorAICallback",
    "AcceleratorAILightningCallback",
    "EngineState",
    "KalmanLossGovernor",
    "ModuleCircuitBreaker",
    "CircuitState",
    "VRAMPressureGuard",
    "PressureLevel",
]
