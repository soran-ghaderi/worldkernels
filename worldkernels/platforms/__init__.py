r"""Hardware-platform abstraction. ``current_platform()`` resolves the active device family."""

from __future__ import annotations

from worldkernels.platforms.cpu import CpuPlatform
from worldkernels.platforms.cuda import CudaPlatform
from worldkernels.platforms.interface import Platform
from worldkernels.platforms.rocm import RocmPlatform

__all__ = ["Platform", "CudaPlatform", "RocmPlatform", "CpuPlatform", "current_platform"]

_PLATFORMS: tuple[type[Platform], ...] = (CudaPlatform, RocmPlatform, CpuPlatform)
_active: Platform | None = None


def current_platform() -> Platform:
    r"""Return the active platform, probing devices once and caching the result."""
    global _active
    if _active is None:
        for platform_cls in _PLATFORMS:
            if platform_cls.is_available():
                _active = platform_cls()
                break
        else:
            _active = CpuPlatform()
    return _active
