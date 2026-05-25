r"""Model management command implementations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def run_list(verbose: bool = False) -> None:
    from worldkernels.worlds.hub import list_models

    hub_models = list_models()

    if not verbose:
        print("Models (use these with --world / --model):")
        for name, card in sorted(hub_models.items()):
            desc = f"  {card.description}" if card.description else ""
            print(f"  {name:30s}{desc}")
        return

    print("Models:")
    for name, card in sorted(hub_models.items()):
        defaults = ", ".join(f"{k}={v}" for k, v in card.default_kwargs.items())
        hf = card.hf_repo or ""
        print(f"  {name:30s}  adapter={card.adapter:16s}  hf={hf:30s}  defaults={{{defaults}}}")

    print()
    print("Adapters (requires torch):")
    from worldkernels.worlds.registry import get_world_class, list_worlds

    for name in list_worlds():
        cls = get_world_class(name)
        mode = getattr(cls, "transition_mode", "unknown")
        streaming = getattr(cls, "supports_streaming", False)
        kv = getattr(cls, "supports_kv_cache", False)
        print(
            f"  {name:20s}  mode={mode.value if hasattr(mode, 'value') else mode}  "
            f"streaming={streaming}  kv_cache={kv}  class={cls.__qualname__}"
        )


def run_inspect(model_id: str, device: str = "cpu", config_json: str | None = None) -> None:
    from worldkernels.core.config import WorldConfig
    from worldkernels.worlds.hub import resolve_model
    from worldkernels.worlds.registry import get_world_class

    adapter_name, _kwargs = resolve_model(model_id)
    cls = get_world_class(adapter_name)

    print(f"Model: {model_id}")
    print(f"Class: {cls.__module__}.{cls.__qualname__}")
    print()

    mode = getattr(cls, "transition_mode", None)
    if mode is not None:
        print(f"Transition mode:    {mode.value if hasattr(mode, 'value') else mode}")
    print(f"Supports streaming: {getattr(cls, 'supports_streaming', False)}")
    print(f"Supports KV cache:  {getattr(cls, 'supports_kv_cache', False)}")

    stage_modes = getattr(cls, "stage_exec_modes", {})
    if stage_modes:
        print()
        print("Stage execution modes:")
        for stage, exec_mode in stage_modes.items():
            s = stage.value if hasattr(stage, "value") else str(stage)
            m = exec_mode.value if hasattr(exec_mode, "value") else str(exec_mode)
            print(f"  {s:12s} -> {m}")

    default_config = getattr(cls, "default_config", None)
    if default_config is not None:
        print()
        print("Default config:")
        for k, v in vars(default_config).items():
            if not k.startswith("_"):
                print(f"  {k}: {v}")

    if config_json:
        cfg = WorldConfig(**json.loads(config_json))
    else:
        cfg = default_config or WorldConfig()

    try:
        import torch

        world = cls()
        world.initialize(device=device, dtype=torch.float32 if device == "cpu" else torch.bfloat16)
        vram = world.profile_vram(cfg)
        print()
        print(
            f"VRAM estimate:      {vram:.1f} MB  (config: {cfg.height}x{cfg.width}, "
            f"steps={cfg.num_inference_steps}, frames_per_step={cfg.frames_per_step})"
        )
    except Exception as exc:
        log.debug("Could not estimate VRAM: %s", exc)


def run_download(model_id: str, revision: str | None = None, cache_dir: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    from worldkernels.worlds.hub import get_model_card

    card = get_model_card(model_id)
    hf_repo = card.hf_repo if card and card.hf_repo else model_id

    dest = cache_dir or None
    print(f"Downloading {hf_repo}" + (f" (revision={revision})" if revision else "") + " ...")
    path = snapshot_download(hf_repo, revision=revision, cache_dir=dest)
    print(f"Downloaded to: {path}")


def run_remove(model_id: str) -> None:
    from huggingface_hub import scan_cache_dir

    from worldkernels.worlds.hub import get_model_card

    card = get_model_card(model_id)
    hf_repo = card.hf_repo if card and card.hf_repo else model_id

    cache_info = scan_cache_dir()
    matched = [repo for repo in cache_info.repos if repo.repo_id == hf_repo]
    if not matched:
        print(f"Model '{hf_repo}' not found in HuggingFace cache.")
        raise SystemExit(1)

    for repo in matched:
        for revision in repo.revisions:
            strategy = cache_info.delete_revisions(revision.commit_hash)
            strategy.execute()
            print(f"Removed {hf_repo} revision {revision.commit_hash[:12]}")

    print(f"Model '{hf_repo}' removed from cache.")


def run_export(
    model_id: str,
    fmt: str = "tensorrt",
    output: str | None = None,
    height: int = 480,
    width: int = 848,
    device: str = "cuda",
) -> None:
    import torch

    from worldkernels.core.config import WorldConfig
    from worldkernels.worlds.hub import resolve_model
    from worldkernels.worlds.registry import get_world_class

    adapter_name, merged_kwargs = resolve_model(model_id)
    cls = get_world_class(adapter_name)
    world = cls(**merged_kwargs)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    world.initialize(device=device, dtype=dtype)

    _cfg = WorldConfig(height=height, width=width)  # noqa: F841
    out_path = Path(output) if output else Path(f"{model_id}_{fmt}")

    if fmt == "tensorrt":
        try:
            import torch_tensorrt  # noqa: F401
        except ImportError:
            print("torch_tensorrt not installed.")
            raise SystemExit(1)

        print(f"Exporting {model_id} to TensorRT at {out_path} ...")
        print("TensorRT export not yet implemented — will be available in a future release.")
    elif fmt == "onnx":
        print(f"Exporting {model_id} to ONNX at {out_path} ...")
        print("ONNX export not yet implemented — will be available in a future release.")
    else:
        print(f"Unknown export format: {fmt}. Supported: tensorrt, onnx")
        raise SystemExit(1)
