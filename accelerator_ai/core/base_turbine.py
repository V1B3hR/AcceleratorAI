"""
BaseTurbine: Abstract base class for all turbine and fluid processing modules.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from accelerator_ai.core.flow_packet import FlowPacket


class TurbineModule(ABC):
    """
    Abstract interface for every component in the VIBE Turbine Architecture.
    
    Each turbine module processes a stream of FlowPackets, altering their pressure,
    entropy, dimensionality, or gradient potential.
    """

    def __init__(self, name: str):
        self.name = name
        self.rpm: float = 0.0
        self.efficiency: float = 1.0
        self.total_processed_packets: int = 0
        self.total_processed_samples: int = 0
        self.last_telemetry: Dict[str, Any] = {}

    @abstractmethod
    def process(self, packet: FlowPacket) -> FlowPacket:
        """
        Main fluid transformation step. Must be implemented by subclasses.
        """
        pass

    def spin(self, delta_rpm: float) -> None:
        """Adjusts the rotor RPM of the turbine module."""
        self.rpm = max(0.0, self.rpm + delta_rpm)

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns the current real-time telemetry snapshot of this turbine module."""
        return {
            "name": self.name,
            "rpm": round(self.rpm, 1),
            "efficiency": round(self.efficiency, 3),
            "processed_packets": self.total_processed_packets,
            "processed_samples": self.total_processed_samples,
            **self.last_telemetry,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', rpm={self.rpm:.1f})>"
