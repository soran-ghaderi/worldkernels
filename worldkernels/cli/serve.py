r"""Serve command implementation."""

from __future__ import annotations

from typing import Any


def run_serve(
    host: str,
    port: int,
    max_sessions: int,
    api_key: str | None,
    device: str,
    model: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> None:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Install with: pip install worldkernels[serve]")
        raise SystemExit(1)

    from worldkernels.core.config import ServerConfig
    from worldkernels.serving.server import create_app

    cfg = ServerConfig(
        host=host,
        port=port,
        max_sessions=max_sessions,
        api_key=api_key,
    )
    app = create_app(cfg, device=device)

    if model is not None:
        from worldkernels.core.engine import WorldKernel

        engine: WorldKernel = app.state.engine
        print(f"Pre-loading model: {model}")
        engine.load_model(model, **(model_kwargs or {}))
        print(f"Model '{model}' ready.")

    uvicorn.run(app, host=host, port=port)
