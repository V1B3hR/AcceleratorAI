"""
BaseTurbine: Abstract base class for all turbine and fluid processing modules.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from accelerator_ai.core.flow_packet import FlowPacket
from accelerator_ai.core.circuit_breaker import ModuleCircuitBreaker


class TurbineModule(ABC):
    """
    Abstract interface for every component in the VIBE Turbine Architecture.
    
    Each turbine module processes a stream of FlowPackets, altering their pressure,
    entropy, dimensionality, or gradient potential. Protected by ModuleCircuitBreaker.
    """

    def __init__(self, name: str, failure_threshold: int = 3, recovery_steps: int = 50):
        self.name = name
        self.rpm: float = 0.0
        self.efficiency: float = 1.0
        self.total_processed_packets: int = 0
        self.total_processed_samples: int = 0
        self.last_telemetry: Dict[str, Any] = {}
        self.circuit_breaker = ModuleCircuitBreaker(
            name=self.name,
            failure_threshold=failure_threshold,
            recovery_steps=recovery_steps,
        )

    @abstractmethod
    def process(self, packet: FlowPacket) -> FlowPacket:
        """
        Main fluid transformation step. Must be implemented by subclasses.
        """
        pass

    def safe_process(self, packet: FlowPacket, current_step: int = 0) -> FlowPacket:
        """
        Executes process() guarded by the module's circuit breaker.
        If the breaker is tripped or process() raises an exception,
        the incoming packet is returned unmodified, preventing training failure.
        """
        return self.circuit_breaker.execute(
            self.process,
            packet,
            fallback=packet,
            current_step=current_step,
        )

    def spin(self, delta_rpm: float) -> None:
        """Adjusts the rotor RPM of the turbine module."""
        self.rpm = max(0.0, self.rpm + delta_rpm)

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns the current real-time telemetry snapshot of this turbine module."""
        return {
            "name": self.name,
            "rpm": round(self.rpm, 1),
            "efficiency": round(self.efficiency, 3),
            "circuit_breaker": self.circuit_breaker.state,
            "processed_packets": self.total_processed_packets,
            "processed_samples": self.total_processed_samples,
            **self.last_telemetry,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', rpm={self.rpm:.1f})>"
