r"""FastAPI application factory for WorldKernels."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from worldkernels.core.config import ServerConfig
from worldkernels.core.engine import WorldKernel
from worldkernels.serving.auth import require_api_key
from worldkernels.serving.routes import configure_routes


def create_app(
    server_config: ServerConfig | None = None,
    device: str = "cuda",
) -> FastAPI:
    r"""Build the FastAPI application with engine and routes wired up."""
    cfg = server_config or ServerConfig()

    app = FastAPI(
        title="WorldKernels",
        description="GPU-first world model simulation engine",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = WorldKernel(
        device=device,
        max_sessions=cfg.max_sessions,
    )
    app.state.engine = engine

    auth_dep = require_api_key(cfg.api_key)
    router = configure_routes(engine, auth_dep=auth_dep)
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        engine.shutdown()

    return app
