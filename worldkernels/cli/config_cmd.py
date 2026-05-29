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

    print(f"{'flag':22s} {'value':10s} source")
    for f in fields:
        print(f"{f:22s} {str(getattr(cfg, f)):10s} {sources.get(f, 'default')}")

    print()
    print(
        f"[nested] scheduler.max_batch_size={cfg.scheduler.max_batch_size}  "
        f"cache.block_frames={cfg.cache.block_frames}  "
        f"parallel.tensor_parallel_size={cfg.parallel.tensor_parallel_size}"
    )
