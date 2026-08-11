r"""Serve command: bootstrap a model and start FastAPI."""

from __future__ import annotations

import os
from typing import Any


def run_serve(
    host: str,
    port: int,
    max_sessions: int,
    api_key: str | None,
    device: str,
    model: str | None = None,
    variant: str | None = None,
    ckpt_path: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    allow_fetch: bool = True,
    quiet: bool = False,
    profile: str | None = None,
    overrides: dict | None = None,
    config_file: str | None = None,
    allowed_origins: list[str] | None = None,
    allowed_methods: list[str] | None = None,
    allowed_headers: list[str] | None = None,
    allow_credentials: bool = False,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
    uvicorn_log_level: str = "info",
    disable_access_log: bool = False,
    root_path: str = "",
) -> None:
    import uvicorn

    from worldkernels import ui
    from worldkernels.bootstrap import ProgressController
    from worldkernels.bootstrap.progress import cli_mode
    from worldkernels.config import resolve_runtime_config
    from worldkernels.core.config import ServerConfig
    from worldkernels.serving.server import create_app

    if quiet:
        os.environ["WORLDKERNELS_QUIET"] = "1"

    runtime_config, sources = resolve_runtime_config(
        profile=profile, cli_overrides=overrides, config_file=config_file
    )
    cfg = ServerConfig(
        host=host,
        port=port,
        max_sessions=max_sessions,
        api_key=api_key,
        allowed_origins=allowed_origins or ["*"],
        allowed_methods=allowed_methods or ["*"],
        allowed_headers=allowed_headers or ["*"],
        allow_credentials=allow_credentials,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        uvicorn_log_level=uvicorn_log_level,  # type: ignore[arg-type]
        disable_access_log=disable_access_log,
        root_path=root_path,
    )
    app = create_app(cfg, device=device, runtime_config=runtime_config)

    if model is not None:
        from worldkernels.engine import WorldEngine

        engine: WorldEngine = app.state.engine
        with ProgressController(mode=cli_mode()) as progress:
            engine.load_model(
                model,
                variant=variant,
                ckpt_path=ckpt_path,
                progress=progress,
                allow_fetch=allow_fetch,
                **(model_kwargs or {}),
            )
            progress.finalize(success=True, summary=f"model={model}")

    print_startup(app, cfg, runtime_config, sources, model=model, device=device)

    uvicorn_kwargs: dict[str, Any] = {
        "log_level": cfg.uvicorn_log_level,
        "access_log": not cfg.disable_access_log,
        "ssl_keyfile": cfg.ssl_keyfile,
        "ssl_certfile": cfg.ssl_certfile,
    }
    try:
        if ui.resolve_output_mode() == ui.OutputMode.AUTO and ui.capabilities().is_terminal:
            _serve_with_dashboard(app, host, port, uvicorn_kwargs)
        else:
            uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
    finally:
        engine = app.state.engine
        ui.info(f"[shutdown] closing {len(engine.list_sessions())} session(s)")
        engine.shutdown()
        ui.info("[shutdown] engine released")


def base_url(cfg: Any) -> str:
    scheme = "https" if cfg.ssl_certfile else "http"
    return f"{scheme}://{cfg.host}:{cfg.port}{cfg.root_path}"


def enumerate_routes(app: Any) -> list[tuple[str, list[str]]]:
    r"""``(path, methods)`` per mounted route; websocket routes report ``WEBSOCKET``.

    Descends into included routers, which newer FastAPI keeps as a nested object
    instead of flattening into ``app.routes``.
    """
    out: list[tuple[str, list[str]]] = []
    _collect_routes(getattr(app, "routes", []), out)
    return out


def _collect_routes(routes: Any, out: list[tuple[str, list[str]]]) -> None:
    for route in routes:
        resolved = getattr(route, "effective_route_contexts", None)
        if callable(resolved):
            _collect_routes(list(resolved()), out)
            continue
        nested = getattr(route, "routes", None) or getattr(
            getattr(route, "original_router", None), "routes", None
        )
        if nested:
            _collect_routes(nested, out)
            continue
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not path:
            inner = getattr(route, "original_route", None) or getattr(
                route, "starlette_route", None
            )
            path = getattr(inner, "path", None)
            methods = methods or getattr(inner, "methods", None)
        if not path:
            continue
        out.append((path, sorted(methods) if methods else ["WEBSOCKET"]))


def print_startup(
    app: Any,
    cfg: Any,
    runtime_config: Any,
    sources: dict[str, str],
    model: str | None,
    device: str,
) -> None:
    from worldkernels import __version__, ui
    from worldkernels.cli.commands.config_cmd import print_runtime_config

    url = base_url(cfg)
    ui.rule(f"worldkernels v{__version__} · serving {model or '(no model preloaded)'}")
    ui.field("device", device)
    ui.field("endpoint", url)
    if cfg.api_key:
        ui.field("auth", "bearer token required")
    print_runtime_config(runtime_config, sources)
    ui.line("Available routes are:")
    for path, methods in enumerate_routes(app):
        ui.line(f"  Route: {path}, Methods: {', '.join(methods)}")
    ui.field("health", f"{url}/health")
    ui.field("metrics", f"{url}/metrics")
    ui.field("docs", f"{url}/docs")


def _serve_with_dashboard(app: Any, host: str, port: int, uvicorn_kwargs: dict[str, Any]) -> None:
    r"""Run uvicorn in a worker thread and a live metrics panel on the main thread."""
    import threading
    import time

    import uvicorn
    from rich.live import Live

    from worldkernels.runtime import metrics
    from worldkernels.ui.console import console

    engine = app.state.engine
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, **{**uvicorn_kwargs, "log_level": "warning"})
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    t0 = time.perf_counter()
    sample = {"t": t0, "steps": metrics.get_steps_total(), "rate": 0.0}

    def _vram() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                metrics.set_vram_bytes(float(total - free))
                return f"{(total - free) / 1e9:.1f} / {total / 1e9:.0f} GB"
        except Exception:
            pass
        return "n/a"

    def render() -> Any:
        from rich.panel import Panel
        from rich.table import Table

        from worldkernels.ui.theme import accent

        now = time.perf_counter()
        el = now - t0
        steps = metrics.get_steps_total()
        dt = now - sample["t"]
        if dt > 0:
            inst = (steps - sample["steps"]) / dt
            sample["rate"] = inst if sample["rate"] == 0.0 else 0.5 * sample["rate"] + 0.5 * inst
            sample["t"], sample["steps"] = now, steps
        sids = engine.list_sessions()

        grid = Table.grid(padding=(0, 3))
        grid.add_row("endpoint", f"http://{host}:{port}")
        grid.add_row("uptime", f"{int(el) // 60:02d}:{int(el) % 60:02d}")
        grid.add_row("sessions", str(len(sids)))
        grid.add_row("steps", f"{int(steps)} total  ·  {sample['rate']:.2f}/s")
        grid.add_row("vram", _vram())
        for sid in sids[:6]:
            s = engine.get_session(sid)
            if s is not None:
                grid.add_row(f"  {sid[:12]}", f"step {s.step_index} · {s.status.value}")
        return Panel(grid, title="worldkernels · serving", border_style=accent(), expand=False)

    try:
        with Live(
            render(), console=console(), screen=True, refresh_per_second=4, transient=False
        ) as live:
            while thread.is_alive():
                live.update(render())
                time.sleep(0.5)
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join(timeout=5)
