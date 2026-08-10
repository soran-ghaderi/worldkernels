r"""Ablation profiles + precedence resolver for RuntimeConfig.

Precedence (lowest to highest): defaults < profile < config file < env (WK_*) < CLI.
``resolve_runtime_config`` returns the resolved config plus a per-field source
map so ``worldkernels config-show`` can attribute every value.
"""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any

from worldkernels.config.cache_config import CacheConfig
from worldkernels.config.parallel_config import ParallelConfig
from worldkernels.config.runtime import (
    ALL_TOGGLE_FIELDS,
    TOGGLE_BOOL_FIELDS,
    TOGGLE_ENUM_FIELDS,
    RuntimeConfig,
)
from worldkernels.config.scheduler_config import SchedulerConfig

__all__ = [
    "PROFILES",
    "NESTED_CONFIGS",
    "CLI_OWNED_FIELDS",
    "resolve_runtime_config",
    "profile_config",
    "split_config_file",
]

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

NESTED_CONFIGS: dict[str, type] = {
    "parallel": ParallelConfig,
    "cache": CacheConfig,
    "scheduler": SchedulerConfig,
}

CLI_OWNED_FIELDS: tuple[str, ...] = ("device", "max_sessions")

_FLAT_FIELDS = {f.name for f in fields(RuntimeConfig)} - set(NESTED_CONFIGS)
_NESTED_FIELDS = {name: {f.name for f in fields(cls)} for name, cls in NESTED_CONFIGS.items()}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def profile_config(name: str) -> RuntimeConfig:
    r"""Materialize a profile into a RuntimeConfig (defaults + profile overrides)."""
    cfg, _ = resolve_runtime_config(profile=name)
    return cfg


def split_config_file(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""Split a YAML config file into ``(frontend, engine)`` mappings.

    Engine keys are RuntimeConfig fields: flat names, dotted paths, or nested
    mappings under ``parallel`` / ``cache`` / ``scheduler``. Everything else
    (host, port, device, ...) is a frontend CLI flag for the caller to apply.
    """
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {str(path)!r} must contain a mapping")
    frontend: dict[str, Any] = {}
    engine: dict[str, Any] = {}
    for key, value in data.items():
        norm = str(key).replace("-", "_")
        root = norm.partition(".")[0]
        if norm in CLI_OWNED_FIELDS or (root not in _FLAT_FIELDS and root not in NESTED_CONFIGS):
            frontend[str(key)] = value
        else:
            engine[norm] = value
    return frontend, engine


def resolve_runtime_config(
    profile: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    env: "os._Environ[str] | dict[str, str] | None" = None,
    config_file: str | Path | None = None,
) -> tuple[RuntimeConfig, dict[str, str]]:
    r"""Resolve a RuntimeConfig under the precedence chain.

    Args:
        profile: Named profile from `PROFILES` (e.g. ``"baseline"``).
        cli_overrides: ``{field: value}`` from CLI flags; nested fields use
            dotted keys (``"parallel.tensor_parallel_size"``). ``None`` values
            are ignored; unknown fields raise ``ValueError``.
        env: Environment mapping (defaults to ``os.environ``).
        config_file: YAML file whose engine keys form the config-file layer
            (frontend keys are ignored here; see ``split_config_file``).

    Returns:
        ``(config, sources)`` where ``sources[field]`` is one of ``"default"``,
        ``"profile:<name>"``, ``"config:<path>"``, ``"env:<VAR>"``,
        ``"cli:--<flag>"``. Nested fields appear under their dotted keys.
    """
    env = os.environ if env is None else env

    cfg = RuntimeConfig()
    sources: dict[str, str] = {f: "default" for f in ALL_TOGGLE_FIELDS}

    if profile:
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choices: {sorted(PROFILES)}")
        for field_name, value in PROFILES[profile].items():
            _set_field(cfg, sources, field_name, value, f"profile:{profile}")

    if config_file is not None:
        _, engine = split_config_file(config_file)
        for field_name, value in _flatten_engine(engine).items():
            _set_field(cfg, sources, field_name, value, f"config:{config_file}")

    for field_name, value, var in _env_overrides(env):
        setattr(cfg, field_name, value)
        sources[field_name] = f"env:{var}"

    for field_name, value in (cli_overrides or {}).items():
        if value is None:
            continue
        flag = field_name.rpartition(".")[2].replace("_", "-")
        _set_field(cfg, sources, field_name, value, f"cli:--{flag}")

    for name in NESTED_CONFIGS:
        getattr(cfg, name).__post_init__()
    return cfg, sources


def _set_field(
    cfg: RuntimeConfig, sources: dict[str, str], name: str, value: Any, source: str
) -> None:
    root, dot, leaf = name.partition(".")
    if dot:
        if root not in NESTED_CONFIGS or leaf not in _NESTED_FIELDS[root]:
            raise ValueError(f"unknown config field {name!r} (from {source})")
        setattr(getattr(cfg, root), leaf, value)
    else:
        if root not in _FLAT_FIELDS:
            raise ValueError(f"unknown config field {name!r} (from {source})")
        allowed = TOGGLE_ENUM_FIELDS.get(root)
        if allowed is not None and value not in allowed:
            raise ValueError(
                f"invalid value {value!r} for {root!r}; choices: {allowed} (from {source})"
            )
        setattr(cfg, root, value)
    sources[name] = source


def _flatten_engine(engine: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in engine.items():
        if key in NESTED_CONFIGS and isinstance(value, dict):
            out.update({f"{key}.{leaf}": v for leaf, v in value.items()})
        else:
            out[key] = value
    return out


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
