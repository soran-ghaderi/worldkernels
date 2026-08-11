r"""Model inspection command implementation."""

from __future__ import annotations

import json
import logging

from worldkernels import ui

log = logging.getLogger(__name__)


def run_inspect(model_id: str, device: str = "cpu", config_json: str | None = None) -> None:
    from worldkernels.core.config import WorldConfig
    from worldkernels.worlds.hub import resolve_model
    from worldkernels.worlds.registry import get_world_class

    adapter_name, _kwargs = resolve_model(model_id)
    cls = get_world_class(adapter_name)

    ui.rule(f"Model: {model_id}")
    ui.field("Class", f"{cls.__module__}.{cls.__qualname__}")

    mode = getattr(cls, "transition_mode", None)
    if mode is not None:
        ui.field("Transition mode", mode.value if hasattr(mode, "value") else mode)
    ui.field("Supports streaming", getattr(cls, "supports_streaming", False))
    ui.field("Supports KV cache", getattr(cls, "supports_kv_cache", False))

    stage_modes = getattr(cls, "stage_exec_modes", {})
    if stage_modes:
        ui.rule("Stage execution modes")
        t = ui.table("stage", "mode")
        for stage, exec_mode in stage_modes.items():
            s = stage.value if hasattr(stage, "value") else str(stage)
            m = exec_mode.value if hasattr(exec_mode, "value") else str(exec_mode)
            t.add_row(s, m)
        ui.print_table(t)

    default_config = getattr(cls, "default_config", None)
    if default_config is not None:
        ui.rule("Default config:")
        for k, v in vars(default_config).items():
            if not k.startswith("_"):
                ui.field(f"  {k}", v)

    if config_json:
        cfg = WorldConfig(**json.loads(config_json))
    else:
        cfg = default_config or WorldConfig()

    try:
        import torch

        world = cls()
        world.initialize(device=device, dtype=torch.float32 if device == "cpu" else torch.bfloat16)
        vram = world.profile_vram(cfg)
        ui.rule()
        ui.ok(
            f"VRAM estimate: {vram:.1f} MB  (config: {cfg.height}x{cfg.width}, "
            f"steps={cfg.num_inference_steps}, frames_per_step={cfg.frames_per_step})"
        )
    except Exception as exc:
        log.debug("Could not estimate VRAM: %s", exc)
