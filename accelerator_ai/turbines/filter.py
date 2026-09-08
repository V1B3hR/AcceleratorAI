"""
Air Filter: Removes noise, NaNs, infinities, and severe outlier particulates.
Prevents damaging the high-speed turbine compressor blades.
"""

from typing import Tuple
import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket


class AirFilter(TurbineModule):
    """
    Simulates high-flow performance air filter.
    Scrubs corrupted data, filters extreme outliers, and purges non-numeric values.
    """

    def __init__(self, outlier_std_threshold: float = 4.0, purge_nans: bool = True):
        super().__init__(name="AirFilter")
        self.outlier_std_threshold = outlier_std_threshold
        self.purge_nans = purge_nans
        self.clog_level: float = 0.0  # 0.0 (clean) to 1.0 (clogged)

    def process(self, packet: FlowPacket) -> FlowPacket:
        """
        Cleans the FlowPacket x and y tensors.
        """
        x = packet.x
        y = packet.y

        # 1. NaN and Inf scrub
        if self.purge_nans:
            x_clean = np.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            y_clean = np.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        else:
            x_clean = x
            y_clean = y

        # 2. Robust outlier particulate detection using Median & MAD
        median = np.median(x_clean, axis=0, keepdims=True)
        mad = np.median(np.abs(x_clean - median), axis=0, keepdims=True) + 1e-5
        # 1.4826 is the normal distribution consistency factor for MAD
        scale = 1.4826 * mad
        z_scores = np.abs((x_clean - median) / scale)

        # Clamping extreme outliers to protect turbine blades
        filtered_x = np.where(
            z_scores > self.outlier_std_threshold,
            median + np.sign(x_clean - median) * (self.outlier_std_threshold * scale),
            x_clean,
        )

        dropped_particles = float(np.sum(z_scores > self.outlier_std_threshold))
        self.clog_level = min(1.0, self.clog_level + (dropped_particles * 0.0001))

        self.total_processed_packets += 1
        self.total_processed_samples += packet.batch_size
        self.efficiency = max(0.5, 1.0 - (self.clog_level * 0.3))

        packet.x = filtered_x
        packet.y = y_clean
        packet.metadata["air_filter_clog"] = round(self.clog_level, 4)
        packet.metadata["scrubbed_particles"] = dropped_particles

        self.last_telemetry = {
            "clog_level": round(self.clog_level, 4),
            "filter_efficiency": round(self.efficiency, 3),
        }
        return packet
