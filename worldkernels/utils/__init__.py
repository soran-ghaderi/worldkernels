r"""Shared utilities: optional-dependency handling and device detection."""

from __future__ import annotations

from worldkernels.utils.device_info import detect_target_device
from worldkernels.utils.import_utils import LazyLoader, PlaceholderModule, optional_import

__all__ = ["PlaceholderModule", "optional_import", "LazyLoader", "detect_target_device"]
