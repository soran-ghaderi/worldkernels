"""Session lifecycle management."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worldkernels.core.config import WorldConfig
    from worldkernels.runtime.executor import Executor
    from worldkernels.worlds.base import AbstractWorld


class SessionStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass
class LatentState:
    r"""Opaque tensor bundle representing the current world state."""

    data: Any = None
    device: str = "cpu"

    @property
    def nbytes(self) -> int:
        r"""Memory footprint in bytes."""
        if self.data is None:
            return 0
        if hasattr(self.data, "nelement") and hasattr(self.data, "element_size"):
            return self.data.nelement() * self.data.element_size()
        return 0

    def clone(self) -> LatentState:
        r"""Deep-copy this state, decoupling from the original tensor."""
        if self.data is None:
            return LatentState(data=None, device=self.device)
        data_copy = self.data.clone() if hasattr(self.data, "clone") else copy.deepcopy(self.data)
        return LatentState(data=data_copy, device=self.device)

    def to(self, device: str) -> LatentState:
        r"""Move state to a different device (e.g. GPU offload to CPU)."""
        if self.data is None or self.device == device:
            return LatentState(data=self.data, device=device)
        moved = self.data.to(device) if hasattr(self.data, "to") else self.data
        return LatentState(data=moved, device=device)


@dataclass
class Session:
    r"""A stateful simulation session bound to a world model."""

    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    world_id: str = ""
    config: WorldConfig | None = None

    # Latent state (kept on GPU or offloaded)
    state: LatentState = field(default_factory=LatentState)
    step_index: int = 0

    # Provenance
    seed: int = 0
    parent_session_id: str | None = None

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    last_active_at: datetime = field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.ACTIVE

    # Checkpoints: checkpoint_id -> LatentState snapshot
    _checkpoints: dict[str, LatentState] = field(default_factory=dict, repr=False)

    # Runtime references (injected by WorldKernel, not serialized)
    _world: AbstractWorld | None = field(default=None, repr=False)
    _executor: Executor | None = field(default=None, repr=False)

    # ---- core hot path ---------------------------------------------------

    def step(
        self,
        action,
        modalities: list[str] | None = None,
        decode: bool = True,
    ):
        r"""Execute one simulation step through the three-stage pipeline.

        Args:
            action: An Action instance describing the control input.
            modalities: Which observation modalities to decode.
                Defaults to ``["frames"]``.
            decode: If False, skip observation decoding (stage 3).

        Returns:
            An Observation with the requested modalities populated.
        """
        from worldkernels.core.errors import SessionPausedError, SessionTerminatedError

        if self.status == SessionStatus.TERMINATED:
            raise SessionTerminatedError(self.session_id)
        if self.status == SessionStatus.PAUSED:
            raise SessionPausedError(self.session_id)

        if self._world is None or self._executor is None:
            raise RuntimeError(
                "Session not bound to a world model. "
                "Create sessions via WorldKernel.create_session()."
            )

        if modalities is None:
            modalities = ["frames"]

        new_state, obs = self._executor.step(
            world=self._world,
            state=self.state,
            action=action,
            modalities=modalities,
            step_index=self.step_index,
            decode=decode,
        )

        self.state = new_state
        self.step_index += 1
        self.last_active_at = datetime.now()
        return obs

    # ---- checkpoint / branch / restore -----------------------------------

    def checkpoint(self) -> str:
        r"""Snapshot current latent state, return checkpoint ID."""
        ckpt_id = f"ckpt_{uuid.uuid4().hex[:8]}"
        self._checkpoints[ckpt_id] = self.state.clone()
        return ckpt_id

    def restore(self, checkpoint_id: str) -> None:
        r"""Restore session to a previously saved checkpoint."""
        if checkpoint_id not in self._checkpoints:
            from worldkernels.core.errors import CheckpointNotFoundError

            raise CheckpointNotFoundError(checkpoint_id, self.session_id)
        self.state = self._checkpoints[checkpoint_id].clone()

    def branch(self) -> Session:
        r"""Clone this session into a new independent session."""
        return Session(
            world_id=self.world_id,
            config=self.config,
            state=self.state.clone(),
            step_index=self.step_index,
            seed=self.seed,
            parent_session_id=self.session_id,
            _world=self._world,
            _executor=self._executor,
        )

    def close(self) -> None:
        self.status = SessionStatus.TERMINATED
        self.state = LatentState()
        self._checkpoints.clear()

    def pause(self) -> None:
        r"""Pause this session (ACTIVE -> PAUSED)."""
        from worldkernels.core.errors import SessionTerminatedError

        if self.status == SessionStatus.TERMINATED:
            raise SessionTerminatedError(self.session_id)
        self.status = SessionStatus.PAUSED

    def resume(self) -> None:
        r"""Resume a paused session (PAUSED -> ACTIVE)."""
        from worldkernels.core.errors import SessionTerminatedError

        if self.status == SessionStatus.TERMINATED:
            raise SessionTerminatedError(self.session_id)
        self.status = SessionStatus.ACTIVE
        self.last_active_at = datetime.now()

    @property
    def id(self) -> str:
        return self.session_id
