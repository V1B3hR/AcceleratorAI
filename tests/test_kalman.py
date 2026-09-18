"""
Unit tests for KalmanLossGovernor (Closed-Loop Kalman Filter for Training Dynamics).
Verifies optimal stochastic noise rejection, true loss and velocity estimation,
plateau detection, divergence detection, and smooth boost modulation.
"""

import pytest
import numpy as np
from accelerator_ai.ecu.kalman import KalmanLossGovernor, KalmanEstimate


def test_kalman_governor_initialization():
    gov = KalmanLossGovernor(initial_loss=2.5)
    assert np.isclose(gov.x[0], 2.5)
    assert np.isclose(gov.x[1], 0.0)
    assert gov.step_count == 0
    assert gov.consecutive_plateau_steps == 0


def test_kalman_noise_rejection_on_steady_descent():
    """
    Simulates a linear descent loss with stochastic measurement noise:
    Loss(t) = 2.0 - 0.01 * t + N(0, 0.05^2)
    Verifies that filtered loss is significantly smoother than raw loss.
    """
    np.random.seed(42)
    gov = KalmanLossGovernor(initial_loss=2.0)

    true_losses = []
    noisy_losses = []
    filtered_losses = []
    velocities = []

    for step in range(1, 51):
        true_loss = max(0.2, 2.0 - 0.02 * step)
        noise = np.random.normal(0.0, 0.05)
        observed = float(true_loss + noise)

        est = gov.update(observed)

        true_losses.append(true_loss)
        noisy_losses.append(observed)
        filtered_losses.append(est.filtered_loss)
        velocities.append(est.loss_velocity)

    # Filtered loss variance around true loss should be lower than noisy observation variance
    noisy_mse = np.mean((np.array(noisy_losses[10:]) - np.array(true_losses[10:])) ** 2)
    filtered_mse = np.mean((np.array(filtered_losses[10:]) - np.array(true_losses[10:])) ** 2)
    assert filtered_mse < noisy_mse, f"Kalman filter did not reduce MSE: {filtered_mse} vs {noisy_mse}"

    # During steady descent, velocity should be negative
    mean_velocity = np.mean(velocities[15:])
    assert mean_velocity < 0.0, f"Expected negative descent velocity, got {mean_velocity}"


def test_kalman_plateau_detection_and_boost_ramp():
    """
    Simulates training stalling on a loss plateau.
    Kalman governor should detect the plateau and recommend elevated boost to overcome stall.
    """
    gov = KalmanLossGovernor(initial_loss=1.0)

    # First feed flat loss for 25 steps
    estimates = []
    for _ in range(25):
        est = gov.update(1.0 + np.random.normal(0.0, 0.0001))
        estimates.append(est)

    last_est = estimates[-1]
    assert last_est.is_plateau is True
    assert last_est.is_diverging is False
    assert last_est.recommended_boost_mod > 1.0, f"Expected boost > 1.0 on plateau, got {last_est.recommended_boost_mod}"
    assert gov.consecutive_plateau_steps > 10


def test_kalman_divergence_detection_and_boost_cut():
    """
    Simulates catastrophic loss explosion / divergence.
    Kalman governor should detect divergence and cut boost ratio.
    """
    gov = KalmanLossGovernor(initial_loss=1.0)

    # 5 steady steps
    for _ in range(5):
        gov.update(1.0)

    # Sudden steep loss spikes
    spike_estimates = []
    loss = 1.0
    for _ in range(5):
        loss += 0.25  # Explosive gradient spike
        est = gov.update(loss)
        spike_estimates.append(est)

    diverging_est = spike_estimates[-1]
    assert diverging_est.is_diverging is True
    assert diverging_est.recommended_boost_mod < 1.0, f"Expected boost cut < 1.0 on divergence, got {diverging_est.recommended_boost_mod}"
    assert diverging_est.loss_velocity > gov.divergence_thresh


def test_kalman_reset():
    gov = KalmanLossGovernor(initial_loss=1.0)
    for _ in range(20):
        gov.update(0.5)

    assert gov.step_count == 20
    gov.reset(initial_loss=3.0)
    assert gov.step_count == 0
    assert np.isclose(gov.x[0], 3.0)
    assert np.isclose(gov.x[1], 0.0)
    assert gov.consecutive_plateau_steps == 0
