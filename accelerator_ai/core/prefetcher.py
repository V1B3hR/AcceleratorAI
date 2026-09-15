"""
CUDAPrefetcher: Asynchronous CUDA stream prefetcher with pinned memory.
Hides Host-to-Device (H2D) PCIe transfer latency by overlapping memory
copies with model computation.
"""

from typing import Iterator, Tuple, Any, Optional
import torch


class CUDAPrefetcher:
    """
    Overlaps CPU-to-GPU data transfers on a dedicated CUDA stream.
    While the GPU trains on batch N, batch N+1 is being copied via DMA.
    """

    def __init__(self, loader: Any, device: Optional[torch.device] = None):
        self.loader = loader
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.iterator = iter(self.loader)
        self.next_x = None
        self.next_y = None
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.next_x = None
            self.next_y = None
            return

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            x, y = batch, None

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                if hasattr(x, "pin_memory") and not x.is_pinned():
                    x = x.pin_memory()
                self.next_x = x.to(self.device, non_blocking=True)
                if y is not None:
                    if hasattr(y, "pin_memory") and not y.is_pinned():
                        y = y.pin_memory()
                    self.next_y = y.to(self.device, non_blocking=True)
                else:
                    self.next_y = None
        else:
            self.next_x = x
            self.next_y = y

    def next(self) -> Tuple[Any, Any]:
        if self.next_x is None:
            return None, None
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        x = self.next_x
        y = self.next_y
        self._preload()
        return x, y

    def __iter__(self):
        return self

    def __next__(self):
        x, y = self.next()
        if x is None:
            raise StopIteration
        return x, y

    def reset(self) -> None:
        """Resets iterator for next epoch."""
        self.iterator = iter(self.loader)
        self._preload()
