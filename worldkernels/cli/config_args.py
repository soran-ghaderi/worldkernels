r"""Dataclass-driven engine flag groups + explicit-override collection."""

from __future__ import annotations

import argparse
import types
import typing
from collections.abc import Iterator
from dataclasses import MISSING, Field, fields, is_dataclass
from typing import Any, Literal

from worldkernels.config.cache_config import CacheConfig
from worldkernels.config.parallel_config import ParallelConfig
from worldkernels.config.profiles import CLI_OWNED_FIELDS
from worldkernels.config.runtime import RuntimeConfig
from worldkernels.config.scheduler_config import SchedulerConfig

__all__ = ["ENGINE_GROUPS", "add_config_group", "add_engine_args", "collect_overrides"]

ENGINE_GROUPS: tuple[tuple[str, type, str], ...] = (
    ("RuntimeConfig", RuntimeConfig, ""),
    ("ParallelConfig", ParallelConfig, "parallel"),
    ("CacheConfig", CacheConfig, "cache"),
    ("SchedulerConfig", SchedulerConfig, "scheduler"),
)


def _unwrap_optional(hint: Any) -> Any:
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def _iter_specs(cls: type, prefix: str) -> Iterator[tuple[Field, str, Any, Any]]:
    hints = typing.get_type_hints(cls)
    for f in fields(cls):
        hint = _unwrap_optional(hints[f.name])
        if f.name in CLI_OWNED_FIELDS or is_dataclass(hint):
            continue
        default = f.default if f.default_factory is MISSING else f.default_factory()
        yield f, (f"{prefix}.{f.name}" if prefix else f.name), hint, default


def add_config_group(
    parser: argparse.ArgumentParser, cls: type, title: str, prefix: str = ""
) -> argparse._ArgumentGroup:
    r"""Add one argparse group per dataclass, one flag per field.

    Bools become ``--flag/--no-flag``, ``Literal`` fields get ``choices``, and
    every flag defaults to ``argparse.SUPPRESS`` so only explicitly-passed
    values appear in the namespace (under dotted dests for nested configs).
    """
    group = parser.add_argument_group(title)
    for f, dest, hint, default in _iter_specs(cls, prefix):
        kwargs: dict[str, Any] = {
            "dest": dest,
            "default": argparse.SUPPRESS,
            "help": f"(default: {default})",
        }
        if hint is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif typing.get_origin(hint) is Literal:
            kwargs["choices"] = typing.get_args(hint)
        else:
            kwargs["metavar"] = f.name.upper()
            if hint in (int, float):
                kwargs["type"] = hint
        group.add_argument(f"--{f.name.replace('_', '-')}", **kwargs)
    return group


def add_engine_args(parser: argparse.ArgumentParser) -> None:
    for title, cls, prefix in ENGINE_GROUPS:
        add_config_group(parser, cls, title, prefix=prefix)


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    r"""Collect explicitly-passed engine flags as ``{field: value}`` with dotted
    keys for nested configs, ready for ``resolve_runtime_config``."""
    out: dict[str, Any] = {}
    for _, cls, prefix in ENGINE_GROUPS:
        for _, dest, _, _ in _iter_specs(cls, prefix):
            if hasattr(args, dest):
                out[dest] = getattr(args, dest)
    return out
