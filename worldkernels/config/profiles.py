r"""Ablation profiles + precedence resolver for RuntimeConfig.

Precedence (lowest to highest): built-in defaults < profile < env (WK_*) < CLI.
``resolve_runtime_config`` returns the resolved config plus a per-field source
map so ``worldkernels config show`` can attribute every value.
"""

from __future__ import annotations

import os
from typing import Any

from worldkernels.config.runtime import (
    ALL_TOGGLE_FIELDS,
    TOGGLE_BOOL_FIELDS,
    TOGGLE_ENUM_FIELDS,
    RuntimeConfig,
)

__all__ = ["PROFILES", "resolve_runtime_config", "profile_config"]

PROFILES: dict[str, dict[str, Any]] = {
    "baseline": {
        "torch_compile": False,
        "cuda_graphs": False,
        "continuous_batching": False,
        "iteration_batching": False,
        "teacache": False,
        "trajectory_cache": False,
        "kv_cache_paged": False,
        "latent_pool": False,
        "offload_idle": False,
        "attention_backend": "sdpa",
    },
    "default": {},
    "fast": {"teacache": True},
    "production": {"teacache": True, "quantization": "int8"},
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def profile_config(name: str) -> RuntimeConfig:
    r"""Materialize a profile into a RuntimeConfig (defaults + profile overrides)."""
    cfg, _ = resolve_runtime_config(profile=name)
    return cfg


def resolve_runtime_config(
    profile: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    env: "os._Environ[str] | dict[str, str] | None" = None,
) -> tuple[RuntimeConfig, dict[str, str]]:
    r"""Resolve a RuntimeConfig under the precedence chain.

    Args:
        profile: Named profile from `PROFILES` (e.g. ``"baseline"``).
        cli_overrides: ``{field: value}`` from CLI flags (``None`` values ignored).
        env: Environment mapping (defaults to ``os.environ``).

    Returns:
        ``(config, sources)`` where ``sources[field]`` is one of ``"default"``,
        ``"profile:<name>"``, ``"env:<VAR>"``, ``"cli:--<flag>"``.
    """
    env = os.environ if env is None else env
    overrides = dict(cli_overrides or {})

    cfg = RuntimeConfig()
    sources: dict[str, str] = {f: "default" for f in ALL_TOGGLE_FIELDS}

    if profile:
        if profile not in PROFILES:
            raise ValueError(
                f"unknown profile {profile!r}; choices: {sorted(PROFILES)}"
            )
        for field_name, value in PROFILES[profile].items():
            setattr(cfg, field_name, value)
            sources[field_name] = f"profile:{profile}"

    for field_name, value, var in _env_overrides(env):
        setattr(cfg, field_name, value)
        sources[field_name] = f"env:{var}"

    for field_name, value in overrides.items():
        if value is None or field_name not in sources:
            continue
        setattr(cfg, field_name, value)
        sources[field_name] = f"cli:--{field_name.replace('_', '-')}"

    return cfg, sources


def _env_overrides(env) -> list[tuple[str, Any, str]]:
    out: list[tuple[str, Any, str]] = []

    disable = env.get("WK_DISABLE", "")
    for name in _split_csv(disable):
        if name in TOGGLE_BOOL_FIELDS:
            out.append((name, False, "WK_DISABLE"))

    enable = env.get("WK_ENABLE", "")
    for name in _split_csv(enable):
        if name in TOGGLE_BOOL_FIELDS:
            out.append((name, True, "WK_ENABLE"))

    for name in TOGGLE_BOOL_FIELDS:
        var = f"WK_{name.upper()}"
        if var in env:
            out.append((name, _coerce_bool(env[var]), var))

    for name, allowed in TOGGLE_ENUM_FIELDS.items():
        var = f"WK_{name.upper()}"
        if var in env:
            val = env[var].strip().lower()
            if val in allowed:
                out.append((name, val, var))

    return out


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _coerce_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return bool(v)
