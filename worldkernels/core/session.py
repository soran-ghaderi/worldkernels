"""Session lifecycle management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worldkernels.core.action import Action
    from worldkernels.core.config import WorldConfig
    from worldkernels.core.observation import Observation


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
    """
    A stateful simulation session.

    Represents an interactive connection to a running world model.
    """

    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    """Unique identifier for this session."""

    world_id: str = ""
    """Which world model this session uses (registry key)."""

    config: WorldConfig | None = None
    """Session configuration (resolution, fps, etc.)."""

    # Latent state (kept on GPU or offloaded)
    state: LatentState = field(default_factory=LatentState)
    """Current latent state of the simulation."""

    step_index: int = 0
    """How many steps have been taken."""

    # Provenance
    seed: int = 0
    action_history: list[Action] = field(default_factory=list)
    """History of actions for replay/debug."""

    parent_session_id: str | None = None
    """If this session was branched from another."""

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    last_active_at: datetime = field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.ACTIVE

    # --- Methods (stubs for now) ---

    def step(self, action: Action) -> Observation:
        """Execute one step with the given action."""
        raise NotImplementedError("Session.step() not yet implemented")

    def checkpoint(self) -> str:
        """Save current state, return checkpoint ID."""
        raise NotImplementedError("Session.checkpoint() not yet implemented")

    def restore(self, checkpoint_id: str) -> None:
        """Restore session to a saved checkpoint."""
        raise NotImplementedError("Session.restore() not yet implemented")

    def branch(self) -> Session:
        """Clone this session's state into a new session."""
        raise NotImplementedError("Session.branch() not yet implemented")

    def close(self) -> None:
        """Terminate this session and release resources."""
        self.status = SessionStatus.TERMINATED

    @property
    def id(self) -> str:
        """Alias for session_id."""
        return self.session_id
