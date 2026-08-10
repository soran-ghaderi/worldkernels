r"""Central registry of WorldKernels environment variables."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from worldkernels.config.runtime import ALL_TOGGLE_FIELDS

__all__ = ["environment_variables", "EXTERNAL_VARIABLES", "set_wk_vars"]


def _read(name: str, default: str | None = None) -> Callable[[], str | None]:
    return lambda: os.environ.get(name, default)


# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    # Comma-separated RuntimeConfig toggle fields to force off (e.g. "teacache,cuda_graphs").
    "WK_DISABLE": _read("WK_DISABLE", ""),
    # Comma-separated RuntimeConfig toggle fields to force on.
    "WK_ENABLE": _read("WK_ENABLE", ""),
    # Hardware target for device-specific dependency selection: "cuda" | "rocm" | "cpu".
    "WK_TARGET_DEVICE": _read("WK_TARGET_DEVICE"),
    # Any value suppresses non-essential CLI output; set automatically by `-q`.
    "WORLDKERNELS_QUIET": _read("WORLDKERNELS_QUIET"),
    # Per-field RuntimeConfig overrides, e.g. WK_TEACACHE=1, WK_DTYPE=bf16, WK_QUANTIZATION=int8.
    **{f"WK_{name.upper()}": _read(f"WK_{name.upper()}") for name in ALL_TOGGLE_FIELDS},
}
# --8<-- [end:env-vars-definition]

EXTERNAL_VARIABLES: tuple[str, ...] = ("NO_COLOR", "CI", "HF_TOKEN", "HF_HOME", "HF_HUB_OFFLINE")


def set_wk_vars() -> dict[str, str]:
    r"""Registry variables currently present in the environment."""
    return {name: os.environ[name] for name in environment_variables if name in os.environ}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
