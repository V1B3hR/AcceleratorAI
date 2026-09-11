"""
Air Filter: Removes noise, NaNs, infinities, and severe outlier particulates.
Prevents damaging the high-speed turbine compressor blades.
"""

from typing import Tuple, Dict, Any
import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket


class AirFilter(TurbineModule):
    """
    Simulates high-flow performance air filter with multi-stage scrubbing:
    1. Mechanical Pleated Mesh: Scrubs NaNs, infinities, and severe univariate MAD outliers.
    2. Neodymium Magnetic Separator: Traps multivariate "heavy metal" nano-particulates
       (adversarial perturbations, poisoned combinations, and high-leverage collinear spikes).
    3. Ultrasonic Sonication Stage:
       - Disperses dense batch clumps and duplicate samples via acoustic micro-dispersion.
       - Fires periodic piezoelectric cleaning pulses to purge filter clogging (sonoclean).
    """

    def __init__(
        self,
        outlier_std_threshold: float = 4.0,
        purge_nans: bool = True,
        fast_mode: bool = True,
        enable_magnetic_stage: bool = True,
        magnetic_threshold_sigma: float = 4.0,
        enable_ultrasonic_stage: bool = True,
        ultrasonic_dedup_threshold: float = 0.98,
        ultrasonic_clean_interval: int = 50,
        ultrasonic_stride: int = 5,
    ):
        super().__init__(name="AirFilter")
        self.outlier_std_threshold = outlier_std_threshold
        self.purge_nans = purge_nans
        self.fast_mode = fast_mode
        self.enable_magnetic_stage = enable_magnetic_stage
        self.magnetic_threshold_sigma = magnetic_threshold_sigma
        self.enable_ultrasonic_stage = enable_ultrasonic_stage
        self.ultrasonic_dedup_threshold = ultrasonic_dedup_threshold
        self.ultrasonic_clean_interval = ultrasonic_clean_interval
        self.ultrasonic_stride = max(1, ultrasonic_stride)
        self.clog_level: float = 0.0  # 0.0 (clean) to 1.0 (clogged)

    def process(self, packet: FlowPacket) -> FlowPacket:
        """
        Cleans the FlowPacket x and y tensors using 3-stage adaptive filtering.
        """
        x = packet.x
        y = packet.y

        # =====================================================================
        # Native PyTorch Tensor Fast Path (CUDA / CPU with Zero Host-Device Sync)
        # =====================================================================
        if hasattr(x, "is_cuda"):
            if not x.is_floating_point():
                # Discrete integer token IDs (LLMs / Transformers): bypass continuous arithmetic scrubbers
                self.total_processed_packets += 1
                self.total_processed_samples += packet.batch_size
                packet.metadata["air_filter_clog"] = round(self.clog_level, 4)
                packet.metadata["scrubbed_particles"] = 0.0
                packet.metadata["magnetic_trapped"] = 0
                packet.metadata["sonicated_clusters"] = 0
                packet.metadata["piezo_cleaned"] = False
                packet.metadata["ultrasonic_threshold"] = round(self.ultrasonic_dedup_threshold, 4)
                return packet
            else:
                import torch
                if self.purge_nans:
                    has_nan_x = bool(torch.isnan(x).any() or torch.isinf(x).any())
                    x_clean = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4) if has_nan_x else x
                    has_nan_y = bool(y is not None and (torch.isnan(y).any() or torch.isinf(y).any())) if (y is not None and y.is_floating_point()) else False
                    y_clean = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4) if (has_nan_y and y is not None) else y
                else:
                    x_clean, y_clean = x, y

                med = torch.quantile(x_clean, 0.5, dim=0, keepdim=True)
                diff = torch.abs(x_clean - med)
                mad = torch.quantile(diff, 0.5, dim=0, keepdim=True) + 1e-5
                scale = 1.4826 * mad
                z_scores = diff / scale
                outliers = z_scores > self.outlier_std_threshold
                if outliers.any():
                    filtered_x = torch.where(outliers, med + torch.sign(x_clean - med) * (self.outlier_std_threshold * scale), x_clean)
                    dropped_particles = float(outliers.sum().item())
                else:
                    filtered_x = x_clean

                self.total_processed_packets += 1
                self.total_processed_samples += packet.batch_size
                packet.x = filtered_x
                packet.y = y_clean
                packet.metadata["air_filter_clog"] = round(self.clog_level, 4)
                packet.metadata["scrubbed_particles"] = dropped_particles
                packet.metadata["magnetic_trapped"] = 0
                packet.metadata["sonicated_clusters"] = 0
                packet.metadata["piezo_cleaned"] = False
                packet.metadata["ultrasonic_threshold"] = round(self.ultrasonic_dedup_threshold, 4)
                return packet

        # =====================================================================
        # Stage 1: Mechanical Paper Mesh (NaN/Inf Purge & Univariate MAD)
        # =====================================================================
        if self.purge_nans:
            has_nan_x = np.isnan(x).any() or np.isinf(x).any()
            has_nan_y = np.isnan(y).any() or np.isinf(y).any()
            x_clean = np.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4) if has_nan_x else x
            y_clean = np.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4) if has_nan_y else y
        else:
            x_clean = x
            y_clean = y

        dropped_particles = 0.0
        n_samples = len(x_clean)
        k = n_samples // 2

        # Robust Median & MAD via O(N) quickselect np.partition
        med = np.partition(x_clean, k, axis=0)[k:k+1]
        diff = np.abs(x_clean - med)
        mad = np.partition(diff, k, axis=0)[k:k+1] + 1e-5
        scale = 1.4826 * mad
        z_scores = diff / scale
        outliers = z_scores > self.outlier_std_threshold

        if outliers.any():
            filtered_x = np.where(
                outliers,
                med + np.sign(x_clean - med) * (self.outlier_std_threshold * scale),
                x_clean,
            )
            dropped_particles = float(np.sum(outliers))
        else:
            filtered_x = x_clean

        # =====================================================================
        # Stage 2: Neodymium Magnetic Separator ("Heavy Metals" / Nano-Particles)
        # Traps multivariate anomalies and subtle adversarial vectors
        # =====================================================================
        magnetic_trapped = 0
        if self.enable_magnetic_stage and n_samples > 3:
            # Scaled multivariate Mahalanobis proxy
            std_residuals = (filtered_x - med) / scale
            multi_dist = np.sqrt(np.mean(std_residuals ** 2, axis=1))  # Root Mean Square Z per sample
            heavy_metal_mask = multi_dist > self.magnetic_threshold_sigma

            if heavy_metal_mask.any():
                magnetic_trapped = int(np.sum(heavy_metal_mask))
                # Soft projection: pull heavy metal particulates back to magnetic safe boundary
                clamp_factors = (self.magnetic_threshold_sigma / np.maximum(multi_dist, 1e-6))
                for idx in np.where(heavy_metal_mask)[0]:
                    filtered_x[idx] = med[0] + (filtered_x[idx] - med[0]) * clamp_factors[idx]

        # =====================================================================
        # Stage 3: Ultrasonic Sonication (De-clustering & Piezo Self-Cleaning)
        # =====================================================================
        sonicated_clusters = 0
        effective_threshold = self.ultrasonic_dedup_threshold
        psi = float(getattr(packet, "pressure", 1.0))

        # Dynamic sensitivity coupling with Information Pressure Psi:
        # High pressure indicates rich, novel informational entropy;
        # dynamically loosen dedup threshold so rare support vectors are preserved.
        if psi > 1.0:
            pressure_relief = min(0.018, (psi - 1.0) * 0.01)
            effective_threshold = min(0.999, self.ultrasonic_dedup_threshold + pressure_relief)

        # Strided screening: execute acoustic checks every N packets or when pressure surges
        should_run_ultrasonic = (
            self.enable_ultrasonic_stage
            and (n_samples > 1)
            and (
                (self.total_processed_packets % self.ultrasonic_stride == 0)
                or (psi >= 2.0)
            )
        )

        if should_run_ultrasonic:
            norms = np.linalg.norm(filtered_x, axis=1, keepdims=True) + 1e-7
            normalized_x = filtered_x / norms

            if n_samples <= 64:
                # Direct vectorized pairwise cosine matrix for small batches
                sim_matrix = np.dot(normalized_x, normalized_x.T)
                np.fill_diagonal(sim_matrix, 0.0)
                tri_upper = np.triu(sim_matrix, k=1)
                high_sim_pairs = np.where(tri_upper > effective_threshold)
                if len(high_sim_pairs[0]) > 0:
                    sonicated_clusters = len(high_sim_pairs[0])
                    for j in np.unique(high_sim_pairs[1]):
                        harmonic_dispersion = np.sin(np.arange(filtered_x.shape[1])) * (0.01 * scale[0])
                        filtered_x[j] += harmonic_dispersion
            else:
                # O(B log B) Strided Random-Projection Screening for large batches (B > 64)
                # Prevents massive O(B^2) pairwise quadratic explosion
                rng = np.random.RandomState(42)
                proj_vec = rng.randn(normalized_x.shape[1])
                proj_vec /= (np.linalg.norm(proj_vec) + 1e-7)
                projections = np.dot(normalized_x, proj_vec)
                sorted_idx = np.argsort(projections)

                candidate_dups = []
                window = 3
                for i in range(len(sorted_idx) - 1):
                    orig_i = sorted_idx[i]
                    for w in range(1, min(window + 1, len(sorted_idx) - i)):
                        orig_j = sorted_idx[i + w]
                        cos_sim = float(np.dot(normalized_x[orig_i], normalized_x[orig_j]))
                        if cos_sim > effective_threshold:
                            candidate_dups.append(orig_j)

                if candidate_dups:
                    unique_dups = np.unique(candidate_dups)
                    sonicated_clusters = len(unique_dups)
                    for j in unique_dups:
                        harmonic_dispersion = np.sin(np.arange(filtered_x.shape[1])) * (0.01 * scale[0])
                        filtered_x[j] += harmonic_dispersion

        # B. Piezo Self-Cleaning Pulse (Sonoclean)
        # Cleans trapped particulates before filter efficiency drops permanently
        self.clog_level = min(1.0, self.clog_level + ((dropped_particles + magnetic_trapped) * 0.0001))
        self.total_processed_packets += 1
        self.total_processed_samples += packet.batch_size

        piezo_cleaned = False
        if self.enable_ultrasonic_stage:
            if (self.total_processed_packets % self.ultrasonic_clean_interval == 0) or (self.clog_level > 0.7):
                self.clog_level *= 0.1  # 90% purge of accumulated sludge
                piezo_cleaned = True

        self.efficiency = max(0.5, 1.0 - (self.clog_level * 0.3))

        packet.x = filtered_x
        packet.y = y_clean
        packet.metadata["air_filter_clog"] = round(self.clog_level, 4)
        packet.metadata["scrubbed_particles"] = dropped_particles
        packet.metadata["magnetic_trapped"] = magnetic_trapped
        packet.metadata["sonicated_clusters"] = sonicated_clusters
        packet.metadata["piezo_cleaned"] = piezo_cleaned
        packet.metadata["ultrasonic_threshold"] = round(effective_threshold, 4)

        self.last_telemetry = {
            "clog_level": round(self.clog_level, 4),
            "filter_efficiency": round(self.efficiency, 3),
            "scrubbed_particles": dropped_particles,
            "magnetic_trapped": magnetic_trapped,
            "sonicated_clusters": sonicated_clusters,
            "piezo_cleaned": piezo_cleaned,
            "ultrasonic_threshold": round(effective_threshold, 4),
        }
        return packet

    def state_dict(self) -> Dict[str, Any]:
        """Serializes filter state for checkpointing."""
        return {
            "clog_level": float(self.clog_level),
            "efficiency": float(self.efficiency),
            "total_processed_packets": int(self.total_processed_packets),
            "total_processed_samples": int(self.total_processed_samples),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores filter state from checkpoint."""
        self.clog_level = float(state_dict.get("clog_level", 0.0))
        self.efficiency = float(state_dict.get("efficiency", 1.0))
        self.total_processed_packets = int(state_dict.get("total_processed_packets", 0))
        self.total_processed_samples = int(state_dict.get("total_processed_samples", 0))

