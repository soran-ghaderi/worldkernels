r"""Shared low-level terminal UI kernel: theme, console, verbosity, components, live displays."""

from __future__ import annotations

import importlib as _importlib

__all__ = [
    "console",
    "capabilities",
    "rule",
    "table",
    "print_table",
    "field",
    "ok",
    "err",
    "info",
    "line",
    "status",
    "kv_rows",
    "metric_spark",
    "banner",
    "GLYPHS",
    "style",
    "set_mode",
    "get_mode",
    "accent",
    "ramp",
    "sparkline",
    "working",
    "Verbosity",
    "OutputMode",
    "get_verbosity",
    "set_verbosity",
    "get_output_mode",
    "set_output_mode",
    "resolve_output_mode",
    "PhaseEvent",
    "PhaseStatus",
    "LiveDisplay",
    "PhaseProgress",
    "StepProgress",
    "configure_logging",
]

_LAZY: dict[str, tuple[str, str]] = {
    "console": ("worldkernels.ui.console", "console"),
    "capabilities": ("worldkernels.ui.console", "capabilities"),
    "rule": ("worldkernels.ui.components", "rule"),
    "table": ("worldkernels.ui.components", "table"),
    "print_table": ("worldkernels.ui.components", "print_table"),
    "field": ("worldkernels.ui.components", "field"),
    "ok": ("worldkernels.ui.components", "ok"),
    "err": ("worldkernels.ui.components", "err"),
    "info": ("worldkernels.ui.components", "info"),
    "line": ("worldkernels.ui.components", "line"),
    "status": ("worldkernels.ui.components", "status"),
    "kv_rows": ("worldkernels.ui.components", "kv_rows"),
    "metric_spark": ("worldkernels.ui.components", "metric_spark"),
    "banner": ("worldkernels.ui.components", "banner"),
    "GLYPHS": ("worldkernels.ui.theme", "GLYPHS"),
    "style": ("worldkernels.ui.theme", "style"),
    "set_mode": ("worldkernels.ui.theme", "set_mode"),
    "get_mode": ("worldkernels.ui.theme", "get_mode"),
    "accent": ("worldkernels.ui.theme", "accent"),
    "ramp": ("worldkernels.ui.theme", "ramp"),
    "sparkline": ("worldkernels.ui.progress", "sparkline"),
    "working": ("worldkernels.ui.progress", "working"),
    "Verbosity": ("worldkernels.ui.verbosity", "Verbosity"),
    "OutputMode": ("worldkernels.ui.verbosity", "OutputMode"),
    "get_verbosity": ("worldkernels.ui.verbosity", "get_verbosity"),
    "set_verbosity": ("worldkernels.ui.verbosity", "set_verbosity"),
    "get_output_mode": ("worldkernels.ui.verbosity", "get_output_mode"),
    "set_output_mode": ("worldkernels.ui.verbosity", "set_output_mode"),
    "resolve_output_mode": ("worldkernels.ui.verbosity", "resolve_output_mode"),
    "PhaseEvent": ("worldkernels.ui.events", "PhaseEvent"),
    "PhaseStatus": ("worldkernels.ui.events", "PhaseStatus"),
    "LiveDisplay": ("worldkernels.ui.live", "LiveDisplay"),
    "PhaseProgress": ("worldkernels.ui.progress", "PhaseProgress"),
    "StepProgress": ("worldkernels.ui.progress", "StepProgress"),
    "configure_logging": ("worldkernels.ui.logging_setup", "configure_logging"),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        val = getattr(_importlib.import_module(module_path), attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
