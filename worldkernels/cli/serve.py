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
) -> None:
    import uvicorn

    from worldkernels.bootstrap import ProgressController
    from worldkernels.config import resolve_runtime_config
    from worldkernels.core.config import ServerConfig
    from worldkernels.serving.server import create_app

    if quiet:
        os.environ["WORLDKERNELS_QUIET"] = "1"

    runtime_config, _ = resolve_runtime_config(profile=profile)
    cfg = ServerConfig(host=host, port=port, max_sessions=max_sessions, api_key=api_key)
    app = create_app(cfg, device=device, runtime_config=runtime_config)

    if model is not None:
        from worldkernels.engine import WorldEngine

        engine: WorldEngine = app.state.engine
        with ProgressController(mode="quiet" if quiet else "plain") as progress:
            engine.load_model(
                model,
                variant=variant,
                ckpt_path=ckpt_path,
                progress=progress,
                allow_fetch=allow_fetch,
                **(model_kwargs or {}),
            )
            progress.finalize(success=True, summary=f"model={model}")

    print(f"  http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port)
