"""Eager PyTorch backend (baseline).

No compilation, no CUDA graphs. Just ``torch.no_grad()`` with the
model's native forward pass. Always available, works on CPU and any GPU.
"""

from __future__ import annotations

import torch


class EagerBackend:
    """Minimal execution backend that runs model stages directly."""

    name = "eager"

    def __init__(self, device: str, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def run(self, fn, *args, **kwargs):
        """Execute *fn* under ``torch.no_grad`` on the configured device."""
        return fn(*args, **kwargs)
