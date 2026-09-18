"""
ModuleCircuitBreaker: Per-module fault isolation and automated recovery.

Implements the classic Resilience4j / Netflix Hystrix Circuit Breaker pattern.
Ensures that a runtime error or numerical instability in any single turbine,
filter, or injector never halts or destabilizes overall model training.
"""

import time
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "CLOSED"        # Normal execution
    OPEN = "OPEN"            # Tripped, isolating module (graceful bypass)
    HALF_OPEN = "HALF_OPEN"  # Trial probe to test module recovery


class ModuleCircuitBreaker:
    """
    Guards a specific turbine component against cascading failures.

    Args:
        name: Human-readable identifier of the guarded module
        failure_threshold: Consecutive failures before tripping to OPEN (default: 3)
        recovery_steps: Steps to remain in OPEN before probing with HALF_OPEN (default: 50)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_steps: int = 50,
    ):
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_steps = max(1, recovery_steps)

        self._state: CircuitState = CircuitState.CLOSED
        self.consecutive_failures: int = 0
        self.total_failures: int = 0
        self.total_bypasses: int = 0
        self.last_failure_step: int = -1
        self.tripped_at_step: int = -1
        self.last_error_message: Optional[str] = None

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def is_tripped(self) -> bool:
        return self._state == CircuitState.OPEN

    def check_recovery(self, current_step: int) -> None:
        """Transitions from OPEN to HALF_OPEN when cooldown steps have elapsed."""
        if self._state == CircuitState.OPEN:
            if (current_step - self.tripped_at_step) >= self.recovery_steps:
                logger.info(
                    "CircuitBreaker [%s]: Recovery cooldown elapsed (%d steps). "
                    "Transitioning to HALF_OPEN trial state.",
                    self.name,
                    self.recovery_steps,
                )
                self._state = CircuitState.HALF_OPEN

    def record_success(self) -> None:
        """Records a successful execution, restoring breaker to CLOSED."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info(
                "CircuitBreaker [%s]: Trial execution succeeded. "
                "Circuit breaker fully reset to CLOSED.",
                self.name,
            )
        self._state = CircuitState.CLOSED
        self.consecutive_failures = 0

    def record_failure(self, err: Exception, current_step: int = 0) -> None:
        """Records an execution failure, tripping to OPEN if threshold exceeded."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_step = current_step
        self.last_error_message = f"{type(err).__name__}: {str(err)}"

        logger.warning(
            "CircuitBreaker [%s] failure #%d/%d at step %d: %s",
            self.name,
            self.consecutive_failures,
            self.failure_threshold,
            current_step,
            self.last_error_message,
        )

        if self.consecutive_failures >= self.failure_threshold or self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self.tripped_at_step = current_step
            logger.error(
                "CircuitBreaker [%s]: FAILURE THRESHOLD EXCEEDED. Tripping breaker to OPEN. "
                "Module will be bypassed for the next %d steps to guarantee cluster survival.",
                self.name,
                self.recovery_steps,
            )

    def execute(
        self,
        fn: Callable[..., Any],
        *args: Any,
        fallback: Any = None,
        current_step: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Executes fn(*args, **kwargs) guarded by the circuit breaker.
        If OPEN, bypasses fn and returns fallback immediately.
        If an exception occurs, records failure and returns fallback safely.
        """
        self.check_recovery(current_step)

        if self._state == CircuitState.OPEN:
            self.total_bypasses += 1
            return fallback

        try:
            result = fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e, current_step=current_step)
            self.total_bypasses += 1
            return fallback

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_bypasses": self.total_bypasses,
            "last_error": self.last_error_message,
        }
