"""
Unit tests for ModuleCircuitBreaker (Per-Module Circuit Breaker & Fault Isolation).
Validates resilience against runtime exceptions, failure counters, fallback bypass,
and automatic recovery cooldown transitions.
"""

import pytest
import numpy as np
from accelerator_ai.core.circuit_breaker import ModuleCircuitBreaker, CircuitState
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket


class CrashingTurbine(TurbineModule):
    """Test turbine that intentionally crashes to verify isolation."""

    def __init__(self, should_crash: bool = True):
        super().__init__(name="CrashingTurbine")
        self.should_crash = should_crash
        self.process_call_count = 0

    def process(self, packet: FlowPacket) -> FlowPacket:
        self.process_call_count += 1
        if self.should_crash:
            raise RuntimeError("Synthetic turbine explosion in compressor blades!")
        packet.pressure *= 1.1
        return packet


def test_circuit_breaker_initial_state():
    cb = ModuleCircuitBreaker(name="TestCompressor", failure_threshold=3, recovery_steps=10)
    assert cb.state == "CLOSED"
    assert not cb.is_tripped
    assert cb.consecutive_failures == 0
    assert cb.total_failures == 0
    assert cb.total_bypasses == 0


def test_circuit_breaker_trips_to_open_after_threshold():
    cb = ModuleCircuitBreaker(name="TestModule", failure_threshold=3, recovery_steps=10)

    def faulty_func():
        raise ValueError("Degraded math")

    # 1st failure
    res1 = cb.execute(faulty_func, fallback="fallback_1", current_step=1)
    assert res1 == "fallback_1"
    assert cb.state == "CLOSED"
    assert cb.consecutive_failures == 1

    # 2nd failure
    res2 = cb.execute(faulty_func, fallback="fallback_2", current_step=2)
    assert res2 == "fallback_2"
    assert cb.state == "CLOSED"
    assert cb.consecutive_failures == 2

    # 3rd failure -> Trips to OPEN
    res3 = cb.execute(faulty_func, fallback="fallback_3", current_step=3)
    assert res3 == "fallback_3"
    assert cb.state == "OPEN"
    assert cb.is_tripped
    assert cb.tripped_at_step == 3


def test_circuit_breaker_bypasses_when_open():
    cb = ModuleCircuitBreaker(name="TestModule", failure_threshold=2, recovery_steps=20)
    
    call_tracker = {"called": False}
    def crashing():
        call_tracker["called"] = True
        raise RuntimeError("Crash")

    # Trip breaker
    cb.execute(crashing, fallback=None, current_step=1)
    cb.execute(crashing, fallback=None, current_step=2)
    assert cb.is_tripped

    # Reset call tracker to confirm crashing function is NOT called while OPEN
    call_tracker["called"] = False
    result = cb.execute(crashing, fallback="safely_bypassed", current_step=5)
    assert result == "safely_bypassed"
    assert call_tracker["called"] is False
    assert cb.total_bypasses == 3


def test_circuit_breaker_recovery_to_half_open_and_closed():
    cb = ModuleCircuitBreaker(name="TestModule", failure_threshold=2, recovery_steps=10)

    def faulty_func():
        raise ZeroDivisionError("Div zero")

    cb.execute(faulty_func, fallback=None, current_step=1)
    cb.execute(faulty_func, fallback=None, current_step=2)
    assert cb.state == "OPEN"

    # Before cooldown: remains OPEN
    cb.check_recovery(current_step=11)  # step delta = 9 < 10
    assert cb.state == "OPEN"

    # Cooldown step reached: transitions to HALF_OPEN
    cb.check_recovery(current_step=12)  # step delta = 10 >= 10
    assert cb.state == "HALF_OPEN"

    # Successful trial execution resets to CLOSED
    def healthy_func():
        return 42

    trial_res = cb.execute(healthy_func, fallback=-1, current_step=13)
    assert trial_res == 42
    assert cb.state == "CLOSED"
    assert cb.consecutive_failures == 0


def test_circuit_breaker_half_open_failure_re_trips_immediately():
    cb = ModuleCircuitBreaker(name="TestModule", failure_threshold=3, recovery_steps=5)

    def faulty_func():
        raise RuntimeError("Persistent fault")

    # Trip to OPEN
    for step in range(1, 4):
        cb.execute(faulty_func, fallback=None, current_step=step)
    assert cb.state == "OPEN"

    # Advance past recovery cooldown
    cb.check_recovery(current_step=10)
    assert cb.state == "HALF_OPEN"

    # Single failure during HALF_OPEN immediately re-trips to OPEN
    cb.execute(faulty_func, fallback="retry_failed", current_step=11)
    assert cb.state == "OPEN"
    assert cb.tripped_at_step == 11


def test_turbine_safe_process_isolation():
    crashing_turbine = CrashingTurbine(should_crash=True)
    initial_packet = FlowPacket(x=np.ones((4, 8)), y=np.zeros(4), pressure=1.5)

    # 1st and 2nd step: turbine raises error, but safe_process swallows it and returns initial_packet
    out1 = crashing_turbine.safe_process(initial_packet, current_step=1)
    assert np.allclose(out1.x, initial_packet.x)
    assert crashing_turbine.circuit_breaker.consecutive_failures == 1

    out2 = crashing_turbine.safe_process(initial_packet, current_step=2)
    assert np.allclose(out2.x, initial_packet.x)
    assert crashing_turbine.circuit_breaker.consecutive_failures == 2

    # 3rd step: trips circuit breaker to OPEN
    out3 = crashing_turbine.safe_process(initial_packet, current_step=3)
    assert np.allclose(out3.x, initial_packet.x)
    assert crashing_turbine.circuit_breaker.is_tripped

    # 4th step: turbine is completely isolated - process() is NOT even called!
    call_count_before = crashing_turbine.process_call_count
    out4 = crashing_turbine.safe_process(initial_packet, current_step=4)
    assert np.allclose(out4.x, initial_packet.x)
    assert crashing_turbine.process_call_count == call_count_before  # Bypass confirmed!
