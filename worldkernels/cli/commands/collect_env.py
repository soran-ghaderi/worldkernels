r"""Collect environment info: GPU, deps, plugins, hub, local cache, isolated envs.

Prints what's useful when filing a bug report or sanity-checking a new install.
"""

from __future__ import annotations

import importlib
import shutil
import sys

from worldkernels import ui

_PY_MIN = (3, 10)
_PY_MAX_EXCL = (3, 14)


def run_collect_env() -> None:
    ui.rule("worldkernels collect-env")

    _check_python_version()

    ui.rule("python & libs")
    _check_import("torch", required=True)
    _check_import("huggingface_hub", required=True)
    _check_import("fastapi", required=True)
    _check_import("rich", required=False)
    _check_import("transformers", required=False)
    _check_import("diffusers", required=False)
    _check_import("flash_attn", required=False)

    ui.rule("platform")
    try:
        from worldkernels.platforms import current_platform
        from worldkernels.utils.device_info import detect_target_device

        platform = current_platform()
        ui.field("target device", detect_target_device())
        ui.field("platform", platform.name)
        ui.field("device count", platform.device_count())
        ui.field("attention backend", platform.default_attention_backend())
        ui.field("default dtype", str(platform.default_dtype()))
    except Exception as exc:
        ui.err(f"platform probe failed: {exc}")

    ui.rule("gpu")
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                free, total = torch.cuda.mem_get_info(i)
                gb_free = free / (1024**3)
                gb_total = total / (1024**3)
                ui.field(
                    f"cuda:{i}",
                    f"{name}  sm_{cap[0]}{cap[1]}  {gb_free:.1f}/{gb_total:.1f} GB free",
                    value_style="green",
                )
        else:
            ui.info("no cuda available")
    except Exception as exc:
        ui.err(f"torch check failed: {exc}")

    ui.rule("tools")
    for tool in ("git", "uv"):
        found = shutil.which(tool)
        ui.field(
            tool,
            "found" if found else "MISSING",
            value_style="green" if found else "red bold",
        )

    ui.rule("hub · registered models")
    try:
        from worldkernels.worlds.hub import list_models

        t = ui.table("model", "adapter")
        for name, card in sorted(list_models().items()):
            t.add_row(name, card.adapter)
        ui.print_table(t)
    except Exception as exc:
        ui.err(f"hub load failed: {exc}")

    ui.rule("local cache")
    try:
        from worldkernels.bootstrap import cache

        manifests = cache.list_manifests()
        ui.field("home", cache.home())
        ui.field("cached models", len(manifests))
        for m in manifests:
            v = f":{m.variant}" if m.variant else ""
            ui.line(f"    {m.model_id}{v}")
    except Exception as exc:
        ui.err(f"cache scan failed: {exc}")

    ui.rule("environment variables · WK_* registry")
    try:
        import os

        from worldkernels import envs as wk_env_registry

        set_vars = wk_env_registry.set_wk_vars()
        for name, value in sorted(set_vars.items()):
            ui.field(name, value, value_style="green")
        for name in wk_env_registry.EXTERNAL_VARIABLES:
            if name in os.environ:
                shown = "***" if name == "HF_TOKEN" else os.environ[name]
                ui.field(name, shown)
        if not set_vars:
            ui.info("(no WK_* variables set)")
    except Exception as exc:
        ui.err(f"env registry failed: {exc}")

    ui.rule("runtime config · defaults, WK_* env applied")
    try:
        from worldkernels.config.profiles import resolve_runtime_config
        from worldkernels.config.runtime import ALL_TOGGLE_FIELDS

        cfg, sources = resolve_runtime_config()
        for f in ALL_TOGGLE_FIELDS:
            src = sources.get(f, "default")
            tag = "" if src == "default" else f"  ({src})"
            ui.field(f, f"{getattr(cfg, f)}{tag}")
    except Exception as exc:
        ui.err(f"runtime config load failed: {exc}")

    ui.rule("isolated envs")
    try:
        from worldkernels.bootstrap import cache
        from worldkernels.runtime import envs

        envs_root = envs.envs_dir()
        if envs_root.exists():
            children = [p for p in envs_root.iterdir() if p.is_dir()]
            ui.field("root", envs_root)
            ui.field("count", len(children))
            for env_dir in children:
                size_mb = cache.directory_size_bytes(env_dir) / (1024 * 1024)
                ui.line(f"    {env_dir.name}  {size_mb:.0f} MB")
        else:
            ui.info("(none)")
    except Exception as exc:
        ui.err(f"env scan failed: {exc}")


def _check_python_version() -> None:
    cur = sys.version_info[:2]
    py = f"{cur[0]}.{cur[1]}.{sys.version_info[2]}"
    supported_range = f">={_PY_MIN[0]}.{_PY_MIN[1]},<{_PY_MAX_EXCL[0]}.{_PY_MAX_EXCL[1]}"
    if cur < _PY_MIN:
        ui.err(
            f"Python {py}  too old; worldkernels supports {supported_range}. "
            "Recreate the venv with `uv venv --python 3.12`."
        )
    elif cur >= _PY_MAX_EXCL:
        ui.err(
            f"Python {py}  ahead of supported range {supported_range}. "
            "Many cosmos/diffusion deps lack cp"
            f"{cur[0]}{cur[1]} wheels; expect source-build failures. "
            "Recreate the venv with `uv venv --python 3.12`."
        )
    else:
        ui.field("Python", f"{py}  (supported {supported_range})", value_style="green")


def _check_import(name: str, required: bool = False) -> None:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        ui.field(name, ver, value_style="green")
    except ImportError:
        if required:
            ui.field(name, "MISSING", value_style="red bold")
        else:
            ui.field(name, "(optional, not installed)", value_style="dim")
