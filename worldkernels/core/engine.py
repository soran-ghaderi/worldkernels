"""WorldKernel main engine class.

Thin coordinator that owns world models, sessions, and the executor.
No heavy compute happens here. The hot path lives in
``runtime.executor.Executor`` and the world adapter stages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from worldkernels.core.errors import (
    SessionLimitError,
    WorldInitError,
    WorldNotFoundError,
)

if TYPE_CHECKING:
    from worldkernels.core.config import WorldConfig
    from worldkernels.core.session import Session
    from worldkernels.worlds.base import AbstractWorld

log = logging.getLogger(__name__)


def _default_dtype(device: str) -> torch.dtype:
    """Pick the best default precision for a device."""
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap >= (8, 0):  # Ampere+
            return torch.bfloat16
        return torch.float16  # Pascal / Turing (GTX 1050 etc.)
    return torch.float32


class WorldKernel:
    """GPU-first world model simulation engine.

    Manages world model loading, session creation, and resource lifecycle.
    This is the main user-facing entry point.

    Example::

        wk = WorldKernel(device="cpu")
        wk.load_world("dummy")
        session = wk.create_session("dummy")
        obs = session.step(Action("keyboard", {"keys": ["W"]}))
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
        # session_id -> Session
        self._sessions: dict[str, Session] = {}

        # Shared executor (one per engine for now)
        from worldkernels.runtime.executor import Executor

        self._executor = Executor(device=device, dtype=self.dtype)

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
        """Load a world model by registry name, HF Hub ID, or local path.

        For now only registry names are supported (e.g. ``"dummy"``).
        HF Hub loading will be added in the diffusers adapter.
        """
        from worldkernels.worlds.registry import get_world_class

        key = alias or model_id.split("/")[-1]

        # Resolve class from registry
        try:
            world_cls = get_world_class(model_id)
        except KeyError:
            raise WorldNotFoundError(model_id)

        # Instantiate and initialize
        world: AbstractWorld = world_cls()
        try:
            world.initialize(device=self.device, dtype=self.dtype)
        except Exception as exc:
            raise WorldInitError(model_id, str(exc)) from exc

        self._worlds[key] = world
        log.info("Loaded world: %s (class=%s)", key, world_cls.__qualname__)

    # ---- session management ----------------------------------------------

    def create_session(
        self,
        world: str,
        config: WorldConfig | None = None,
        seed: int | None = None,
    ) -> Session:
        """Create a new simulation session bound to a loaded world model."""
        from worldkernels.core.config import WorldConfig as WC
        from worldkernels.core.session import Session

        if world not in self._worlds:
            raise WorldNotFoundError(world)

        if len(self._sessions) >= self.max_sessions:
            raise SessionLimitError(self.max_sessions)

        cfg = config or WC()
        actual_seed = seed if seed is not None else 0

        world_instance = self._worlds[world]
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
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        """Terminate a session and release its resources."""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def list_worlds(self) -> list[str]:
        """List all loaded world model aliases."""
        return list(self._worlds.keys())

    def shutdown(self) -> None:
        """Cleanup all sessions and resources."""
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        self._worlds.clear()
        log.info("WorldKernel shut down.")
