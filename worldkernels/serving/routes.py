r"""REST API routes for WorldKernels serving layer."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from worldkernels.core.config import WorldConfig
from worldkernels.core.engine import WorldKernel
from worldkernels.core.errors import (
    SessionLimitError,
    SessionNotFoundError,
    SessionTerminatedError,
    VRAMExhaustedError,
    WorldAlreadyLoadedError,
    WorldInitError,
    WorldKernelError,
    WorldNotFoundError,
)

router = APIRouter(prefix="/v1")


def _engine() -> WorldKernel:
    raise RuntimeError("Engine dependency not configured.")


def configure_routes(engine: WorldKernel, auth_dep: Any = None) -> APIRouter:
    r"""Bind the engine instance and optional auth to the router."""

    deps = [Depends(lambda: engine)]
    if auth_dep is not None:
        deps.append(Depends(auth_dep))

    router.dependency_overrides_provider = None  # type: ignore[attr-defined]

    @router.get("/worlds", tags=["worlds"])
    async def list_worlds() -> dict[str, Any]:
        return {"worlds": engine.list_worlds()}

    @router.post("/worlds", tags=["worlds"], status_code=201)
    async def load_model(req: LoadModelRequest) -> dict[str, str]:
        try:
            await asyncio.to_thread(
                engine.load_model,
                req.model_id,
                alias=req.alias,
                trust_remote_code=req.trust_remote_code,
                **req.kwargs,
            )
        except WorldAlreadyLoadedError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except WorldNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except WorldInitError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        key = req.alias or req.model_id.split("/")[-1]
        return {"status": "loaded", "world": key}

    @router.delete("/worlds/{world_id}", tags=["worlds"])
    async def unload_model(world_id: str) -> dict[str, str]:
        try:
            engine.unload_model(world_id)
        except WorldNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"status": "unloaded", "world": world_id}

    @router.get("/sessions", tags=["sessions"])
    async def list_sessions() -> dict[str, Any]:
        sessions = []
        for sid in engine.list_sessions():
            sess = engine.get_session(sid)
            if sess is not None:
                sessions.append(_session_summary(sess))
        return {"sessions": sessions}

    @router.post("/sessions", tags=["sessions"], status_code=201)
    async def create_session(req: CreateSessionRequest) -> dict[str, Any]:
        config = WorldConfig(
            height=req.height,
            width=req.width,
            fps=req.fps,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            frames_per_step=req.frames_per_step,
            initial_prompt=req.initial_prompt,
        )
        try:
            sess = await asyncio.to_thread(
                engine.create_session, req.world, config=config, seed=req.seed
            )
        except WorldNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except SessionLimitError as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except VRAMExhaustedError as exc:
            raise HTTPException(status_code=507, detail=str(exc))
        return _session_summary(sess)

    @router.get("/sessions/{session_id}", tags=["sessions"])
    async def get_session(session_id: str) -> dict[str, Any]:
        sess = engine.get_session(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        return _session_summary(sess)

    @router.delete("/sessions/{session_id}", tags=["sessions"])
    async def delete_session(session_id: str) -> dict[str, str]:
        sess = engine.get_session(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        engine.close_session(session_id)
        return {"status": "terminated", "session_id": session_id}

    @router.post("/sessions/{session_id}/step", tags=["sessions"])
    async def step(session_id: str, req: StepRequest) -> dict[str, Any]:
        from worldkernels.core.action import Action

        sess = engine.get_session(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        action = Action(action_type=req.action_type, payload=req.payload)
        try:
            obs = await asyncio.to_thread(
                sess.step, action, modalities=req.modalities, decode=req.decode
            )
        except SessionTerminatedError as exc:
            raise HTTPException(status_code=410, detail=str(exc))
        except WorldKernelError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return _observation_to_dict(obs, session_id, sess.step_index)

    @router.post("/sessions/{session_id}/checkpoint", tags=["sessions"])
    async def checkpoint(session_id: str) -> dict[str, str]:
        sess = engine.get_session(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        ckpt_id = sess.checkpoint()
        return {"checkpoint_id": ckpt_id, "session_id": session_id}

    @router.post("/sessions/{session_id}/restore", tags=["sessions"])
    async def restore(session_id: str, req: RestoreRequest) -> dict[str, str]:
        sess = engine.get_session(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        try:
            sess.restore(req.checkpoint_id)
        except WorldKernelError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"status": "restored", "checkpoint_id": req.checkpoint_id}

    @router.post("/sessions/{session_id}/branch", tags=["sessions"], status_code=201)
    async def branch(session_id: str) -> dict[str, Any]:
        sess = engine.get_session(session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        branched = sess.branch()
        engine._sessions[branched.session_id] = branched
        return _session_summary(branched)

    return router


# ---- request / response models -------------------------------------------


class LoadModelRequest(BaseModel):
    model_id: str
    alias: str | None = None
    trust_remote_code: bool = False
    kwargs: dict[str, Any] = Field(default_factory=dict)


class CreateSessionRequest(BaseModel):
    world: str
    height: int = 480
    width: int = 848
    fps: int = 24
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    frames_per_step: int = 8
    initial_prompt: str | None = None
    seed: int | None = None


class StepRequest(BaseModel):
    action_type: str = "null"
    payload: dict[str, Any] = Field(default_factory=dict)
    modalities: list[str] = Field(default_factory=lambda: ["frames"])
    decode: bool = True


class RestoreRequest(BaseModel):
    checkpoint_id: str


# ---- helpers --------------------------------------------------------------


def _session_summary(sess: Any) -> dict[str, Any]:
    return {
        "session_id": sess.session_id,
        "world_id": sess.world_id,
        "status": sess.status.value,
        "step_index": sess.step_index,
        "seed": sess.seed,
        "parent_session_id": sess.parent_session_id,
        "created_at": sess.created_at.isoformat(),
    }


def _observation_to_dict(obs: Any, session_id: str, step_index: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "session_id": session_id,
        "step_index": obs.step_index,
        "generation_time_ms": obs.generation_time_ms,
        "decode_skipped": obs.decode_skipped,
    }
    if obs.frames is not None:
        result["frames"] = [
            base64.b64encode(f).decode("ascii") if isinstance(f, (bytes, bytearray)) else None
            for f in obs.frames
        ]
        result["num_frames"] = len(obs.frames)
    if obs.stage_timing is not None:
        result["stage_timing"] = obs.stage_timing.as_dict()
    return result
