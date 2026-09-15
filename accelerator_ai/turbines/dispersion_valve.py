"""
SwirlDispersionValve: Multi-hole atomization and swirl distribution manifold.

Replaces crude batch concatenation with physical charge atomization and toroidal
swirl mixing. Disperses asynchronous injection packets (synthetic, real-world, chaos shock)
uniformly throughout the intake charge, eliminating gradient knocking and localized shocks.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from accelerator_ai.core.base_turbine import TurbineModule
from accelerator_ai.core.flow_packet import FlowPacket


class SwirlDispersionValve(TurbineModule):
    """
    Simulates a variable-geometry swirl & tumble dispersion valve.
    
    Functions:
      1. Atomization: Disperses high-entropy synthetic / chaos perturbations as a fine
         micro-droplet field across the main batch features (diffusive augmentation).
      2. Stratified Swirl Mixing: Interleaves high-pressure real-world samples in an
         Archimedean swirl sequence, preventing localized gradient clustering.
      3. Homogeneity Regulation: Dynamically tunes dispersion pattern to maintain uniform
         combustion energy without cylinder hot-spots.
    """

    def __init__(
        self,
        name: str = "SwirlDispersionValve",
        atomization_ratio: float = 0.65,    # 65% diffused into existing samples, 35% interleaved
        swirl_angle_deg: float = 45.0,       # Swirl vane blade angle
        diffusion_gamma: float = 0.15,       # Micro-droplet perturbation intensity
        atomization_radius: Optional[float] = None,
        diffusive_mode: bool = True,
        stratified_swirl: bool = True,
        **kwargs: Any,
    ):
        super().__init__(name=name)
        if atomization_radius is not None:
            diffusion_gamma = atomization_radius
        self.atomization_ratio = atomization_ratio
        self.swirl_angle_deg = swirl_angle_deg
        self.diffusion_gamma = diffusion_gamma
        self.atomization_radius = diffusion_gamma
        self.diffusive_mode = diffusive_mode
        self.stratified_swirl = stratified_swirl

        self.last_homogeneity_pct: float = 100.0
        self.last_droplet_fineness: float = 12.5  # Microns analog
        self.total_dispersions: int = 0

    def process(self, packet: FlowPacket) -> FlowPacket:
        """Passthrough placeholder for pipeline chaining."""
        return packet

    def disperse_and_mix(
        self,
        main_packet: FlowPacket,
        injected_packets: Optional[List[FlowPacket]],
    ) -> FlowPacket:
        """
        Atomizes and swirl-distributes injected packets into the main charge.
        
        Returns:
            A homogeneous, perfectly mixed FlowPacket with optimal flame front.
        """
        if not injected_packets or len(injected_packets) == 0:
            self.last_homogeneity_pct = 100.0
            return main_packet

        valid_injections = [p for p in injected_packets if p is not None and p.batch_size > 0]
        if not valid_injections:
            return main_packet

        self.total_dispersions += 1
        is_torch = type(main_packet.x).__module__.startswith("torch")
        if is_torch:
            import torch
            if hasattr(main_packet.x, "is_floating_point") and not main_packet.x.is_floating_point():
                # Fast-Path for Discrete Integer Tokens (LLMs / Transformers):
                # Preserves integer token IDs without float corruption or CPU roundtrips
                discrete_tensors_x = [p.x for p in valid_injections if type(p.x).__module__.startswith("torch")]
                if discrete_tensors_x:
                    cat_x = torch.cat([main_packet.x] + discrete_tensors_x, dim=0)
                    if main_packet.y is not None:
                        cat_y = torch.cat([main_packet.y] + [p.y for p in valid_injections if p.y is not None and type(p.y).__module__.startswith("torch")], dim=0)
                    else:
                        cat_y = None
                else:
                    cat_x, cat_y = main_packet.x, main_packet.y

                self.last_homogeneity_pct = 100.0
                return FlowPacket(
                    x=cat_x,
                    y=cat_y,
                    pressure=main_packet.pressure,
                    viscosity=main_packet.viscosity,
                    temperature=main_packet.temperature,
                    phase=main_packet.phase,
                    source="swirl_atomized_tokens",
                    metadata={"homogeneity_pct": 100.0, "droplet_fineness_um": 10.0},
                )

            # PyTorch Floating Point Tensor fast path (continuous embeddings / features):
            # Pure GPU execution with zero host copies or numpy conversions
            device = main_packet.x.device
            x_main = main_packet.x.clone()
            y_main = main_packet.y.clone() if main_packet.y is not None else None
            n_main = main_packet.batch_size

            diffusive_packets = []
            discrete_packets = []
            for p in valid_injections:
                is_diffusive = (
                    self.diffusive_mode and (
                        p.source in ("synthetic_injector", "shock_injector")
                        or getattr(p, "metadata", {}).get("injector_type") == "synthetic"
                    )
                )
                if is_diffusive:
                    diffusive_packets.append(p)
                else:
                    discrete_packets.append(p)

            if diffusive_packets:
                diff_tensors_x = [
                    p.x.to(device=device, dtype=x_main.dtype) if type(p.x).__module__.startswith("torch")
                    else torch.as_tensor(p.x, device=device, dtype=x_main.dtype)
                    for p in diffusive_packets
                ]
                diff_x = torch.cat(diff_tensors_x, dim=0)
                n_diff = diff_x.size(0)
                swirl_indices = torch.randint(0, n_diff, (n_main,), device=device)
                swirl_factor = float(np.sin(np.radians(self.swirl_angle_deg)))
                effective_gamma = self.diffusion_gamma * swirl_factor
                diff_noise = (diff_x[swirl_indices] - x_main) * effective_gamma
                x_main = x_main + diff_noise
                self.last_droplet_fineness = float(max(2.0, 20.0 - (swirl_factor * 12.0)))

            if discrete_packets:
                disc_tensors_x = [
                    p.x.to(device=device, dtype=x_main.dtype) if type(p.x).__module__.startswith("torch")
                    else torch.as_tensor(p.x, device=device, dtype=x_main.dtype)
                    for p in discrete_packets
                ]
                disc_x = torch.cat(disc_tensors_x, dim=0)
                final_x = torch.cat([x_main, disc_x], dim=0)
                if y_main is not None:
                    disc_tensors_y = [
                        p.y.to(device=device) if type(p.y).__module__.startswith("torch")
                        else torch.as_tensor(p.y, device=device)
                        for p in discrete_packets if p.y is not None
                    ]
                    if disc_tensors_y:
                        final_y = torch.cat([y_main] + disc_tensors_y, dim=0)
                    else:
                        final_y = y_main
                else:
                    final_y = None
            else:
                final_x = x_main
                final_y = y_main

            self.last_homogeneity_pct = 95.0
            total_injected = sum(p.batch_size for p in valid_injections)
            mass_ratio_injected = total_injected / (n_main + total_injected)
            avg_inj_pressure = float(np.mean([p.pressure for p in valid_injections]))
            avg_inj_temp = float(np.mean([p.temperature for p in valid_injections]))
            blended_pressure = float((1.0 - mass_ratio_injected) * main_packet.pressure + mass_ratio_injected * avg_inj_pressure)
            blended_temp = float((1.0 - mass_ratio_injected) * main_packet.temperature + mass_ratio_injected * avg_inj_temp * 0.8)

            return FlowPacket(
                x=final_x,
                y=final_y,
                pressure=blended_pressure,
                viscosity=main_packet.viscosity,
                temperature=blended_temp,
                phase=main_packet.phase,
                source="swirl_atomized_torch",
                metadata={"homogeneity_pct": 95.0, "droplet_fineness_um": self.last_droplet_fineness},
            )

        x_main = main_packet.x.copy()
        y_main = main_packet.y.copy() if main_packet.y is not None else None

        n_main = main_packet.batch_size

        # Accumulate injected fuel components
        diffusive_packets: List[FlowPacket] = []
        discrete_packets: List[FlowPacket] = []

        for p in valid_injections:
            # Check if diffusive atomization is active for synthetic / shock packets
            is_diffusive = (
                self.diffusive_mode and (
                    p.source in ("synthetic_injector", "shock_injector")
                    or getattr(p, "metadata", {}).get("injector_type") == "synthetic"
                )
            )
            if is_diffusive:
                diffusive_packets.append(p)
            else:
                discrete_packets.append(p)

        # -------------------------------------------------------------------
        # 1. Atomization & Feature Diffusion (Mgiełka mikro-kropli w batchu)
        # -------------------------------------------------------------------
        if diffusive_packets:
            # Combine all diffusive packets
            diff_x = np.concatenate([p.x for p in diffusive_packets], axis=0)
            diff_y = np.concatenate([p.y for p in diffusive_packets], axis=0)
            n_diff = len(diff_x)

            # Atomization mapping: disperse across main samples
            # Each main sample receives a swirl-weighted micro-droplet from diff_x
            swirl_indices = np.random.choice(n_diff, size=n_main, replace=True)
            
            # Swirl turbulence weight modulated by vane angle
            swirl_factor = np.sin(np.radians(self.swirl_angle_deg))
            effective_gamma = self.diffusion_gamma * swirl_factor

            # Atomized feature diffusion
            diff_noise = (diff_x[swirl_indices] - x_main) * effective_gamma
            x_main = x_main + diff_noise

            # Soft label smoothing diffusion
            if np.issubdtype(y_main.dtype, np.integer):
                y_main_soft = y_main.astype(np.float32)
            else:
                y_main_soft = y_main.copy()

            label_diff = (diff_y[swirl_indices].astype(np.float32) - y_main_soft) * (effective_gamma * 0.5)
            y_main = np.clip(y_main_soft + label_diff, 0.0, 1.0)

            # Droplet fineness metric (higher gamma / swirl = finer droplets)
            self.last_droplet_fineness = float(max(2.0, 20.0 - (swirl_factor * 12.0)))

        # -------------------------------------------------------------------
        # 2. Stratified Swirl Interleaving (Zawirowanie próbek w batchu)
        # -------------------------------------------------------------------
        if discrete_packets:
            # Collect discrete real-world packets
            disc_x = np.concatenate([p.x for p in discrete_packets], axis=0)
            disc_y = np.concatenate([p.y for p in discrete_packets], axis=0)
            n_disc = len(disc_x)

            if self.stratified_swirl:
                # Toroidal swirl interleaving: vectorized non-blocking distribution
                total_size = n_main + n_disc
                step_interval = max(1, total_size // (n_disc + 1))

                # Vectorized slot assignment without python loops
                disc_target_indices = np.arange(0, total_size, step_interval)[:n_disc]
                disc_mask = np.zeros(total_size, dtype=bool)
                disc_mask[disc_target_indices] = True
                main_mask = ~disc_mask

                interleaved_x = np.empty((total_size, x_main.shape[1]), dtype=x_main.dtype)
                interleaved_y = np.empty(total_size, dtype=y_main.dtype)

                interleaved_x[disc_mask] = disc_x[:len(disc_target_indices)]
                interleaved_y[disc_mask] = disc_y[:len(disc_target_indices)]
                interleaved_x[main_mask] = x_main[:np.sum(main_mask)]
                interleaved_y[main_mask] = y_main[:np.sum(main_mask)]

                final_x = interleaved_x
                final_y = interleaved_y
            else:
                final_x = np.concatenate([x_main, disc_x], axis=0)
                final_y = np.concatenate([y_main, disc_y], axis=0)
        else:
            final_x = x_main
            final_y = y_main

        # -------------------------------------------------------------------
        # 3. Calculate Mixture Properties
        # -------------------------------------------------------------------
        # Homogeneity Index: variance of sample energies across the mixture
        feature_vars = np.var(final_x, axis=1)
        homogeneity = float(np.clip(100.0 - (np.std(feature_vars) * 15.0), 40.0, 99.5))
        self.last_homogeneity_pct = homogeneity

        # Blended physical parameters
        total_injected = sum(p.batch_size for p in valid_injections)
        mass_ratio_injected = total_injected / (n_main + total_injected)
        
        avg_inj_pressure = np.mean([p.pressure for p in valid_injections])
        avg_inj_temp = np.mean([p.temperature for p in valid_injections])

        blended_pressure = float((1.0 - mass_ratio_injected) * main_packet.pressure + mass_ratio_injected * avg_inj_pressure)
        blended_temp = float((1.0 - mass_ratio_injected) * main_packet.temperature + mass_ratio_injected * avg_inj_temp * 0.8) # Cooling from atomization

        self.last_telemetry = {
            "homogeneity_pct": round(self.last_homogeneity_pct, 1),
            "droplet_fineness_um": round(self.last_droplet_fineness, 1),
            "atomized_packets": len(diffusive_packets),
            "interleaved_samples": sum(p.batch_size for p in discrete_packets),
            "swirl_angle_deg": self.swirl_angle_deg,
        }

        return FlowPacket(
            x=final_x,
            y=final_y,
            pressure=blended_pressure,
            viscosity=main_packet.viscosity,
            temperature=blended_temp,
            phase=main_packet.phase,
            source="swirl_atomized_mix",
            metadata={
                "homogeneity_pct": round(self.last_homogeneity_pct, 1),
                "droplet_fineness_um": round(self.last_droplet_fineness, 1),
            },
        )
