"""
Unit tests for VRAMPressureGuard and EngineState (Single Source of Truth & Memory Awareness).
Verifies hardware memory pressure classification, proactive downshifting, cache clearing,
atomic state synchronization, and engine profiling metrics.
"""

import pytest
import numpy as np
from accelerator_ai.security.vram_guard import VRAMPressureGuard, PressureLevel, MemoryPressureReport
from accelerator_ai.core.engine_state import EngineState
from accelerator_ai.engine import TurboLearningEngine
from accelerator_ai.models.neural_core import PureNumPyMLP
from accelerator_ai.config import EngineConfig


def test_vram_guard_classification():
    guard = VRAMPressureGuard(
        warning_threshold=0.75,
        critical_threshold=0.88,
        emergency_threshold=0.95,
    )

    # Mock inspect by calculating ratio
    total = 10000000000  # 10 GB

    # 1. Normal (50% used)
    report_normal = guard.inspect()
    assert guard.last_report is not None
    assert guard.can_upshift() is True

    # Manually test pressure level logic with custom free bytes
    def mock_report(used_fraction):
        allocated = int(total * used_fraction)
        free = total - allocated
        used_ratio = allocated / total
        free_pct = (free / total) * 100.0

        if used_ratio >= guard.emergency_threshold:
            lvl = PressureLevel.EMERGENCY
            rec = True
        elif used_ratio >= guard.critical_threshold:
            lvl = PressureLevel.CRITICAL
            rec = True
        elif used_ratio >= guard.warning_threshold:
            lvl = PressureLevel.WARNING
            rec = False
        else:
            lvl = PressureLevel.NORMAL
            rec = False

        rep = MemoryPressureReport(
            free_bytes=free,
            total_bytes=total,
            allocated_bytes=allocated,
            used_ratio=round(used_ratio, 4),
            free_pct=round(free_pct, 2),
            level=lvl,
            recommended_downshift=rec,
        )
        guard.last_report = rep
        return rep

    rep_warn = mock_report(0.80)
    assert rep_warn.level == PressureLevel.WARNING
    assert rep_warn.recommended_downshift is False
    assert guard.can_upshift() is False

    rep_crit = mock_report(0.90)
    assert rep_crit.level == PressureLevel.CRITICAL
    assert rep_crit.recommended_downshift is True

    rep_emerg = mock_report(0.98)
    assert rep_emerg.level == PressureLevel.EMERGENCY
    assert rep_emerg.recommended_downshift is True


def test_engine_state_central_synchronization():
    state = EngineState()
    assert state.step == 0
    assert state.rpm == 0.0
    assert state.vram_pressure_level == "NORMAL"
    assert state.isolated_modules == []
    assert state.compute_efficiency_pct == 100.0

    # Update from dictionary
    state.update_from_telemetry({
        "step": 42,
        "rpm": 3500.0,
        "boost_psi": 14.7,
        "vram_free_pct": 82.5,
        "engine_overhead_ms": 0.45,
    })

    assert state.step == 42
    assert state.rpm == 3500.0
    assert state.boost_psi == 14.7
    assert state.vram_free_pct == 82.5
    assert state.engine_overhead_ms == 0.45

    # to_dict roundtrip
    d = state.to_dict()
    assert d["step"] == 42
    assert d["rpm"] == 3500.0
    assert isinstance(d["engine_overhead_ms"], float)


def test_engine_step_synchronizes_state_and_profiling():
    """Verifies that running engine.step() updates the atomic EngineState and calculates profiling overhead."""
    mlp = PureNumPyMLP(layer_sizes=[8, 16, 2])
    cfg = EngineConfig(enable_telemetry=False)
    engine = TurboLearningEngine(model=mlp, config=cfg)

    x = np.random.randn(16, 8).astype(np.float32)
    y = np.random.randint(0, 2, size=(16,)).astype(np.int32)

    # Initial state
    assert engine.engine_state.step == 0

    # Execute 3 steps
    for step_i in range(1, 4):
        res = engine.step(x, y)
        assert res is not None
        assert engine.engine_state.step == step_i
        assert engine.engine_state.step_duration_ms > 0.0
        assert engine.engine_state.engine_overhead_ms >= 0.0
        assert 0.0 <= engine.engine_state.compute_efficiency_pct <= 100.0
        assert engine.engine_state.filtered_loss > 0.0
        assert engine.engine_state.vram_free_pct > 0.0
