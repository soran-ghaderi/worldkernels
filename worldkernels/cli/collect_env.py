r"""Collect environment info: GPU, deps, plugins, hub, local cache, isolated envs.

Mirrors `vllm collect-env` / `python -m torch.utils.collect_env`: prints what's
useful when filing a bug report or sanity-checking a new install.
"""

from __future__ import annotations

import importlib
import shutil


def run_collect_env() -> None:
    print("worldkernels collect-env\n")

    print("Python & libs:")
    _check_import("torch", required=True)
    _check_import("huggingface_hub", required=True)
    _check_import("fastapi", required=True)
    _check_import("rich", required=False)
    _check_import("transformers", required=False)
    _check_import("diffusers", required=False)
    _check_import("flash_attn", required=False)

    print("\nGPU:")
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                free, total = torch.cuda.mem_get_info(i)
                gb_free = free / (1024**3)
                gb_total = total / (1024**3)
                print(f"  cuda:{i}  {name}  sm_{cap[0]}{cap[1]}  {gb_free:.1f}/{gb_total:.1f} GB free")
        else:
            print("  no cuda available")
    except Exception as exc:
        print(f"  torch check failed: {exc}")

    print("\nTools:")
    print(f"  git: {'found' if shutil.which('git') else 'MISSING'}")
    print(f"  uv:  {'found' if shutil.which('uv') else 'MISSING'}")

    print("\nHub (registered models):")
    try:
        from worldkernels.worlds.hub import list_models

        for name, card in sorted(list_models().items()):
            print(f"  {name:32s}  adapter={card.adapter}")
    except Exception as exc:
        print(f"  hub load failed: {exc}")

    print("\nLocal cache:")
    try:
        from worldkernels.bootstrap import cache

        manifests = cache.list_manifests()
        print(f"  home: {cache.home()}")
        print(f"  cached models: {len(manifests)}")
        for m in manifests:
            v = f":{m.variant}" if m.variant else ""
            print(f"    {m.model_id}{v}")
    except Exception as exc:
        print(f"  cache scan failed: {exc}")

    print("\nIsolated envs:")
    try:
        from worldkernels.bootstrap import cache
        from worldkernels.runtime import envs

        envs_root = envs.envs_dir()
        if envs_root.exists():
            children = [p for p in envs_root.iterdir() if p.is_dir()]
            print(f"  root: {envs_root}")
            print(f"  count: {len(children)}")
            for env_dir in children:
                size_mb = cache.directory_size_bytes(env_dir) / (1024 * 1024)
                print(f"    {env_dir.name}  {size_mb:.0f} MB")
        else:
            print("  (none)")
    except Exception as exc:
        print(f"  env scan failed: {exc}")


def _check_import(name: str, required: bool = False) -> None:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  {name:18s} {ver}")
    except ImportError:
        tag = "MISSING" if required else "(optional, not installed)"
        print(f"  {name:18s} {tag}")
