r"""`worldkernels config-show` — print the resolved runtime config + source of each flag."""

from __future__ import annotations


def run_config_show(profile: str | None = None, as_json: bool = False) -> None:
    from worldkernels.config.profiles import resolve_runtime_config
    from worldkernels.config.runtime import ALL_TOGGLE_FIELDS

    cfg, sources = resolve_runtime_config(profile=profile)
    fields = ["device", "max_sessions", *ALL_TOGGLE_FIELDS]

    if as_json:
        import json

        data = {f: {"value": getattr(cfg, f), "source": sources.get(f, "default")} for f in fields}
        print(json.dumps(data, indent=2))
        return

    from worldkernels import ui

    ui.rule(f"runtime config · {profile or 'default'}")
    t = ui.table("flag", "value", "source")
    for f in fields:
        src = sources.get(f, "default")
        t.add_row(f, str(getattr(cfg, f)), "" if src == "default" else f"[green]{src}[/green]")
    ui.print_table(t)

    ui.rule("nested")
    ui.field("scheduler.max_batch_size", cfg.scheduler.max_batch_size)
    ui.field("cache.block_frames", cfg.cache.block_frames)
    ui.field("parallel.tensor_parallel_size", cfg.parallel.tensor_parallel_size)
