r"""WebSocket frame streaming.

A client opens ``/v1/sessions/{id}/stream``, then sends one JSON action per
message and receives one JSON observation back. Long-lived sessions avoid the
per-step HTTP handshake and let the engine pipeline steps.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from worldkernels.engine.async_engine import AsyncEngine

__all__ = ["register_websocket_routes"]


def _observation_payload(obs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step_index": obs.step_index,
        "generation_time_ms": obs.generation_time_ms,
        "decode_skipped": obs.decode_skipped,
    }
    if obs.frames is not None:
        payload["frames"] = [
            base64.b64encode(f).decode("ascii") if isinstance(f, (bytes, bytearray)) else None
            for f in obs.frames
        ]
    if obs.stage_timing is not None:
        payload["stage_timing"] = obs.stage_timing.as_dict()
    return payload


def register_websocket_routes(router: APIRouter, async_engine: "AsyncEngine") -> None:
    r"""Attach the session-streaming WebSocket route to ``router``."""

    @router.websocket("/sessions/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        from worldkernels.core.action import Action
        from worldkernels.core.errors import WorldKernelError

        await websocket.accept()
        if async_engine.engine.get_session(session_id) is None:
            await websocket.send_json({"error": f"session {session_id!r} not found"})
            await websocket.close()
            return
        try:
            while True:
                message = await websocket.receive_json()
                action = Action(
                    action_type=message.get("action_type", "null"),
                    payload=message.get("payload", {}),
                )
                try:
                    obs = await async_engine.step(
                        session_id,
                        action,
                        modalities=message.get("modalities", ["frames"]),
                        decode=message.get("decode", True),
                    )
                except WorldKernelError as exc:
                    await websocket.send_json({"error": str(exc)})
                    continue
                await websocket.send_json(_observation_payload(obs))
        except WebSocketDisconnect:
            return
