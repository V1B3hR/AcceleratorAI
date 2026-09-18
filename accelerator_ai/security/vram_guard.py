"""
VRAMPressureGuard: Proactive Memory Pressure Awareness and Preemptive OOM Defense.

Monitors real-time GPU/Host memory headroom before forward/backward steps,
preventing Out-Of-Memory exceptions by pre-emptively modulating VVT gearbox
ratios and flushing transient cache buffers.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class PressureLevel(Enum):
    NORMAL = "NORMAL"          # < 75% allocated - full performance
    WARNING = "WARNING"        # 75% - 88% allocated - hold current gear, prevent upshifts
    CRITICAL = "CRITICAL"      # 88% - 95% allocated - preemptively downshift gear, flush cache
    EMERGENCY = "EMERGENCY"    # > 95% allocated - force Gear 1 immediately, aggressive cache purge


@dataclass
class MemoryPressureReport:
    free_bytes: int
    total_bytes: int
    allocated_bytes: int
    used_ratio: float
    free_pct: float
    level: PressureLevel
    recommended_downshift: bool


class VRAMPressureGuard:
    """
    Proactively audits device memory pressure to eliminate OutOfMemoryError.

    Args:
        warning_threshold: Fraction of memory used to trigger WARNING (default: 0.75)
        critical_threshold: Fraction of memory used to trigger CRITICAL (default: 0.88)
        emergency_threshold: Fraction of memory used to trigger EMERGENCY (default: 0.95)
        check_interval: Steps between hardware device queries (default: 1)
    """

    def __init__(
        self,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.88,
        emergency_threshold: float = 0.95,
        check_interval: int = 1,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.check_interval = max(1, check_interval)

        self.last_report: Optional[MemoryPressureReport] = None
        self.total_preemptive_flushes: int = 0
        self.total_preemptive_downshifts: int = 0

    def inspect(self, step: int = 0) -> MemoryPressureReport:
        """
        Inspects current device memory pressure.
        Uses torch.cuda.mem_get_info() for fast on-device query without synchronization stalls.
        """
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            else:
                # CPU fallback: Assume plenty of headroom
                free_bytes, total_bytes = 8 * (1024 ** 3), 16 * (1024 ** 3)
        except Exception:
            free_bytes, total_bytes = 8 * (1024 ** 3), 16 * (1024 ** 3)

        allocated_bytes = total_bytes - free_bytes
        used_ratio = allocated_bytes / max(1, total_bytes)
        free_pct = (free_bytes / max(1, total_bytes)) * 100.0

        if used_ratio >= self.emergency_threshold:
            level = PressureLevel.EMERGENCY
            rec_downshift = True
        elif used_ratio >= self.critical_threshold:
            level = PressureLevel.CRITICAL
            rec_downshift = True
        elif used_ratio >= self.warning_threshold:
            level = PressureLevel.WARNING
            rec_downshift = False
        else:
            level = PressureLevel.NORMAL
            rec_downshift = False

        report = MemoryPressureReport(
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            allocated_bytes=allocated_bytes,
            used_ratio=round(used_ratio, 4),
            free_pct=round(free_pct, 2),
            level=level,
            recommended_downshift=rec_downshift,
        )
        self.last_report = report
        return report

    def can_upshift(self) -> bool:
        """Returns True only if device memory is in NORMAL state."""
        if self.last_report is None:
            return True
        return self.last_report.level == PressureLevel.NORMAL

    def remedy_if_critical(self) -> bool:
        """
        Checks if memory is in CRITICAL or EMERGENCY state.
        If so, flushes CUDA cache preemptively before training step forward pass.
        Returns True if cache flush was executed.
        """
        if self.last_report and self.last_report.level in (PressureLevel.CRITICAL, PressureLevel.EMERGENCY):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.total_preemptive_flushes += 1
                    logger.warning(
                        "VRAMPressureGuard: Preemptively flushed CUDA cache (Free: %.1f%%, Level: %s).",
                        self.last_report.free_pct,
                        self.last_report.level.value,
                    )
                    return True
            except Exception:
                pass
        return False
