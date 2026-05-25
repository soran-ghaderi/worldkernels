r"""FastAPI application factory for WorldKernels."""

from __future__ import annotations

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from worldkernels.config import ServerConfig
from worldkernels.engine import AsyncEngine, WorldEngine
from worldkernels.serving.auth import require_api_key
from worldkernels.serving.routes import configure_routes


def create_app(
    server_config: ServerConfig | None = None,
    device: str = "cuda",
) -> FastAPI:
    r"""Build the FastAPI application with engine, routes, and metrics wired up."""
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

    engine = WorldEngine(device=device, max_sessions=cfg.max_sessions)
    async_engine = AsyncEngine(engine)
    app.state.engine = engine
    app.state.async_engine = async_engine

    router = configure_routes(async_engine, auth_dep=require_api_key(cfg.api_key))
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        from worldkernels.runtime import metrics as wk_metrics

        return Response(content=wk_metrics.render(), media_type=wk_metrics.CONTENT_TYPE_LATEST)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await async_engine.shutdown()

    return app
