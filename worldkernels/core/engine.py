"""WorldKernel main engine class."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from worldkernels.core.errors import (
    SessionLimitError,
    VRAMExhaustedError,
    WorldAlreadyLoadedError,
    WorldInitError,
    WorldNotFoundError,
)

if TYPE_CHECKING:
    from worldkernels.core.config import WorldConfig
    from worldkernels.core.session import Session
    from worldkernels.worlds.base import AbstractWorld

log = logging.getLogger(__name__)


def _default_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap >= (8, 0):
            return torch.bfloat16
        return torch.float16
    return torch.float32


class WorldKernel:
    r"""GPU-first world model simulation engine.

    Args:
        device: Target device ("cuda", "cpu").
        max_sessions: Maximum concurrent sessions.
        offload_idle: Whether to offload idle session state.
    """

    def __init__(
        self,
        device: str = "cuda",
        max_sessions: int = 4,
        offload_idle: bool = True,
    ) -> None:
        self.device = device
        self.max_sessions = max_sessions
        self.offload_idle = offload_idle
        self.dtype = _default_dtype(device)

        # world_alias -> instantiated AbstractWorld
        self._worlds: dict[str, AbstractWorld] = {}
        self._sessions: dict[str, Session] = {}

        # Shared executor (one per engine for now)
        from worldkernels.runtime.executor import Executor

        self._executor = Executor(
            device=device,
            dtype=self.dtype,
        )

        log.info(
            "WorldKernel initialized: device=%s, dtype=%s, max_sessions=%d",
            device, self.dtype, max_sessions,
        )

    # ---- world loading ---------------------------------------------------

    def load_world(
        self,
        model_id: str,
        alias: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        r"""Load a world model by registry name or HF Hub ID."""
        from worldkernels.core.config import WorldConfig as WC
        from worldkernels.worlds.registry import get_world_class

        key = alias or model_id.split("/")[-1]

        if key in self._worlds:
            raise WorldAlreadyLoadedError(key)

        try:
            world_cls = get_world_class(model_id)
        except KeyError:
            raise WorldNotFoundError(model_id)

        world: AbstractWorld = world_cls()
        try:
            world.initialize(device=self.device, dtype=self.dtype)
        except Exception as exc:
            raise WorldInitError(model_id, str(exc)) from exc

        world.warmup(world.default_config or WC())

        self._worlds[key] = world
        log.info("Loaded world: %s (class=%s)", key, world_cls.__qualname__)

    # ---- session management ----------------------------------------------

    def create_session(
        self,
        world: str,
        config: WorldConfig | None = None,
        seed: int | None = None,
    ) -> Session:
        r"""Create a new simulation session bound to a loaded world model."""
        from worldkernels.core.config import WorldConfig as WC
        from worldkernels.core.session import Session

        if world not in self._worlds:
            raise WorldNotFoundError(world)

        if len(self._sessions) >= self.max_sessions:
            raise SessionLimitError(self.max_sessions)

        cfg = config or WC()
        actual_seed = seed if seed is not None else 0

        world_instance = self._worlds[world]

        vram_mb = world_instance.estimate_vram_mb(cfg)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
            if vram_mb > free_mb:
                raise VRAMExhaustedError(required_mb=vram_mb, available_mb=free_mb)

        initial_state = world_instance.create_initial_state(cfg, actual_seed)

        session = Session(
            world_id=world,
            config=cfg,
            state=initial_state,
            seed=actual_seed,
            _world=world_instance,
            _executor=self._executor,
        )
        self._sessions[session.session_id] = session
        log.info("Created session: %s (world=%s)", session.session_id, world)
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def list_worlds(self) -> list[str]:
        return list(self._worlds.keys())

    def unload_world(self, name: str) -> None:
        r"""Unload a world model and close all its sessions."""
        if name not in self._worlds:
            raise WorldNotFoundError(name)
        sessions_to_close = [
            sid for sid, sess in self._sessions.items()
            if sess.world_id == name
        ]
        for sid in sessions_to_close:
            self.close_session(sid)
        del self._worlds[name]
        log.info("Unloaded world: %s", name)

    def shutdown(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        self._worlds.clear()
        log.info("WorldKernel shut down.")
