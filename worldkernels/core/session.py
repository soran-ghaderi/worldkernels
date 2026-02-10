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
    """Lifecycle status of a session."""

    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass
class LatentState:
    """Opaque tensor bundle representing the current world state."""

    data: Any = None  # Will hold GPU tensors when implemented
    device: str = "cpu"


@dataclass
class Session:
    """A stateful simulation session bound to a world model."""

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
    ):
        """Execute one simulation step with the given action.

        Args:
            action: An Action instance describing the control input.
            modalities: Which observation modalities to decode.
                Defaults to ``["frames"]``. Pass ``["frames", "depth",
                "audio"]`` to decode everything.

        Returns:
            An Observation with the requested modalities populated.
        """
        from worldkernels.core.errors import SessionTerminatedError

        if self.status == SessionStatus.TERMINATED:
            raise SessionTerminatedError(self.session_id)

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
        )

        self.state = new_state
        self.step_index += 1
        self.last_active_at = datetime.now()
        return obs

    # ---- checkpoint / branch / restore -----------------------------------

    def checkpoint(self) -> str:
        """Snapshot current latent state, return checkpoint ID."""
        import torch

        ckpt_id = f"ckpt_{uuid.uuid4().hex[:8]}"
        # Deep copy the tensor data so mutations don't affect the snapshot
        data_copy = self.state.data.clone() if hasattr(self.state.data, "clone") else copy.deepcopy(self.state.data)
        self._checkpoints[ckpt_id] = LatentState(
            data=data_copy, device=self.state.device
        )
        return ckpt_id

    def restore(self, checkpoint_id: str) -> None:
        """Restore session to a previously saved checkpoint."""
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Checkpoint '{checkpoint_id}' not found.")
        saved = self._checkpoints[checkpoint_id]
        data_copy = saved.data.clone() if hasattr(saved.data, "clone") else copy.deepcopy(saved.data)
        self.state = LatentState(data=data_copy, device=saved.device)

    def branch(self) -> Session:
        """Clone this session into a new independent session.

        The new session shares the same world model and executor but
        gets its own copy of the latent state (copy-on-write in the
        future, eager clone for now).
        """
        import torch

        data_copy = self.state.data.clone() if hasattr(self.state.data, "clone") else copy.deepcopy(self.state.data)
        new_session = Session(
            world_id=self.world_id,
            config=self.config,
            state=LatentState(data=data_copy, device=self.state.device),
            step_index=self.step_index,
            seed=self.seed,
            parent_session_id=self.session_id,
            _world=self._world,
            _executor=self._executor,
        )
        return new_session

    def close(self) -> None:
        """Terminate this session and release resources."""
        self.status = SessionStatus.TERMINATED
        self.state = LatentState()  # release tensor reference
        self._checkpoints.clear()

    @property
    def id(self) -> str:
        """Alias for session_id."""
        return self.session_id
