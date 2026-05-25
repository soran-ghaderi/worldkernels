r"""Quantization registry.

Maps a `QuantizationScheme` to an applier that quantizes a module's
weights in place. ``none`` is identity; ``int8`` / ``int4`` weight-only
quantization is delegated to ``torchao`` and fails with an actionable install
message when that optional dependency is absent.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable

from worldkernels.utils import optional_import

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = ["QuantizationScheme", "QuantizationRegistry"]


class QuantizationScheme(str, Enum):
    r"""Supported weight-quantization schemes."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"


def _apply_none(module: "nn.Module") -> "nn.Module":
    return module


def _apply_int8(module: "nn.Module") -> "nn.Module":
    ao = optional_import("torchao", "quant")
    ao.quantization.quantize_(module, ao.quantization.int8_weight_only())
    return module


def _apply_int4(module: "nn.Module") -> "nn.Module":
    ao = optional_import("torchao", "quant")
    ao.quantization.quantize_(module, ao.quantization.int4_weight_only())
    return module


class QuantizationRegistry:
    r"""Resolves a quantization scheme to its weight applier."""

    def __init__(self) -> None:
        self._appliers: dict[QuantizationScheme, Callable[["nn.Module"], "nn.Module"]] = {
            QuantizationScheme.NONE: _apply_none,
            QuantizationScheme.INT8: _apply_int8,
            QuantizationScheme.INT4: _apply_int4,
        }

    def register(
        self,
        scheme: QuantizationScheme,
        applier: Callable[["nn.Module"], "nn.Module"],
    ) -> None:
        r"""Register or override the applier for ``scheme``."""
        self._appliers[scheme] = applier

    def apply(self, module: "nn.Module", scheme: QuantizationScheme | str) -> "nn.Module":
        r"""Quantize ``module`` in place under ``scheme`` and return it."""
        scheme = QuantizationScheme(scheme)
        applier = self._appliers.get(scheme)
        if applier is None:
            raise ValueError(f"no applier registered for scheme {scheme.value!r}")
        return applier(module)
