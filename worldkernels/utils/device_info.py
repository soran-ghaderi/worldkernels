r"""Hardware target detection for device-specific dependency selection."""

from __future__ import annotations

import os
from typing import Literal

__all__ = ["detect_target_device"]

TargetDevice = Literal["cuda", "rocm", "cpu"]


def detect_target_device() -> TargetDevice:
    r"""Return the hardware target.

    Honors the ``WK_TARGET_DEVICE`` environment variable; otherwise probes
    torch for CUDA / HIP, falling back to ``"cpu"``. Does not import torch
    unless the env var is unset.
    """
    forced = os.environ.get("WK_TARGET_DEVICE")
    if forced in ("cuda", "rocm", "cpu"):
        return forced  # type: ignore[return-value]
    try:
        import torch
    except ImportError:
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None) and torch.cuda.is_available():
        return "cuda"
    return "cpu"
