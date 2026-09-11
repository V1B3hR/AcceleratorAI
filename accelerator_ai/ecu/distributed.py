"""
Distributed Master-ECU Coordinator for AcceleratorAI.
Ensures cluster-wide lockstep synchronization across multi-GPU / DDP / FSDP setups.
"""

from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger("accelerator_ai.distributed")


class DistributedECUCoordinator:
    """
    Coordinates engine states across distributed workers in PyTorch DDP / FSDP.

    In distributed training, non-deterministic or divergent micro-batch sizing (VVT),
    shock injections, or wastegate clipping across different ranks causes immediate
    gradient reduction deadlocks and NCCL timeouts.

    Architecture (Master-ECU):
    - Rank 0 acts as the Master ECU, evaluating thermodynamic pressure, VVT gear,
      learning rate, and relief decisions.
    - Packs decisions into a compact 4-float tensor:
      [vvt_gear, learning_rate, wastegate_open_flag, shock_active_flag]
    - Performs an asynchronous or synchronous single-collective broadcast to all worker ranks.
    - All ranks adopt the exact same gear and execution path in lockstep.
    - In standalone mode (single GPU / CPU), operates with zero overhead.
    """

    def __init__(self, master_rank: int = 0):
        self.master_rank = master_rank
        self._is_distributed: bool = False
        self._rank: int = 0
        self._world_size: int = 1
        self._check_distributed_status()

    def _check_distributed_status(self) -> None:
        """Inspects if PyTorch distributed communication group is active."""
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self._is_distributed = True
                self._rank = dist.get_rank()
                self._world_size = dist.get_world_size()
                logger.info(
                    "DistributedECUCoordinator initialized on rank %d/%d (Master=%d)",
                    self._rank,
                    self._world_size,
                    self.master_rank,
                )
            else:
                self._is_distributed = False
                self._rank = 0
                self._world_size = 1
        except Exception:
            self._is_distributed = False
            self._rank = 0
            self._world_size = 1

    @property
    def is_distributed(self) -> bool:
        """Returns True if training inside a multi-node / multi-GPU distributed group."""
        return self._is_distributed

    @property
    def is_master(self) -> bool:
        """Returns True if the current process is the Master ECU (Rank 0)."""
        return (not self._is_distributed) or (self._rank == self.master_rank)

    @property
    def rank(self) -> int:
        """Current process rank index."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Total number of participating distributed processes."""
        return self._world_size

    def broadcast_engine_state(
        self,
        vvt_gear: int,
        learning_rate: float,
        wastegate_open: bool,
        shock_fired: bool,
        device: Optional[Any] = None,
    ) -> Tuple[int, float, bool, bool]:
        """
        Synchronizes engine dynamics across all distributed ranks.
        Rank 0 broadcasts its decision tensor; all worker ranks receive and adopt it.

        Args:
            vvt_gear: Desired VVT Gear index (1, 2, 3).
            learning_rate: Current learning rate calculated by Braided ECU.
            wastegate_open: True if wastegate relief was tripped.
            shock_fired: True if entropy shock nozzle fired.
            device: Optional torch.device for communication buffer.

        Returns:
            Tuple of (synchronized_gear, synchronized_lr, synchronized_wastegate, synchronized_shock).
        """
        if not self._is_distributed:
            return vvt_gear, learning_rate, wastegate_open, shock_fired

        try:
            import torch
            import torch.distributed as dist

            if device is None:
                if torch.cuda.is_available():
                    device = torch.device(f"cuda:{torch.cuda.current_device()}")
                else:
                    device = torch.device("cpu")

            if self.is_master:
                state_tensor = torch.tensor(
                    [
                        float(vvt_gear),
                        float(learning_rate),
                        1.0 if wastegate_open else 0.0,
                        1.0 if shock_fired else 0.0,
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                state_tensor = torch.zeros(4, dtype=torch.float32, device=device)

            # Master-to-workers collective broadcast
            dist.broadcast(state_tensor, src=self.master_rank)

            sync_gear = int(round(state_tensor[0].item()))
            sync_lr = float(state_tensor[1].item())
            sync_wastegate = bool(state_tensor[2].item() > 0.5)
            sync_shock = bool(state_tensor[3].item() > 0.5)

            return sync_gear, sync_lr, sync_wastegate, sync_shock
        except Exception as e:
            logger.warning("DistributedECUCoordinator broadcast failed: %s. Falling back to local state.", e)
            return vvt_gear, learning_rate, wastegate_open, shock_fired

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns distributed coordinator telemetry."""
        return {
            "is_distributed": self._is_distributed,
            "rank": self._rank,
            "world_size": self._world_size,
            "is_master": self.is_master,
        }
