"""WorldKernel main engine class."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worldkernels.core.config import WorldConfig
    from worldkernels.core.session import Session


class WorldKernel:
    """
    GPU-first world model simulation engine.

    This is the main entry point for the WorldKernels library.
    It manages world model loading, session creation, and resource lifecycle.

    Example:
        >>> from worldkernels import WorldKernel, WorldConfig
        >>> wk = WorldKernel(device="cuda")
        >>> wk.load_world("Etched/oasis-500m")
        >>> session = wk.create_session(world="oasis-500m", config=WorldConfig())
        >>> obs = session.step(Action("keyboard", {"keys": ["W"]}))
    """

    def __init__(
        self,
        device: str = "cuda",
        max_sessions: int = 4,
        offload_idle: bool = True,
    ) -> None:
        """
        Initialize the WorldKernel engine.

        Args:
            device: Device to run on ('cuda', 'cuda:0', 'cpu').
            max_sessions: Maximum concurrent sessions.
            offload_idle: Whether to offload paused sessions to CPU.
        """
        self.device = device
        self.max_sessions = max_sessions
        self.offload_idle = offload_idle
        self._worlds: dict[str, object] = {}
        self._sessions: dict[str, Session] = {}

    def load_world(
        self,
        model_id: str,
        alias: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        """
        Load a world model from HuggingFace Hub or local path.

        Args:
            model_id: HuggingFace model ID (e.g., 'Etched/oasis-500m') or local path.
            alias: Optional alias for referencing this world.
            trust_remote_code: Whether to trust remote code from the model repo.
        """
        # Stub implementation
        key = alias or model_id.split("/")[-1]
        self._worlds[key] = {"model_id": model_id, "loaded": False}

    def create_session(
        self,
        world: str,
        config: WorldConfig | None = None,
        seed: int | None = None,
    ) -> Session:
        """
        Create a new simulation session.

        Args:
            world: Name/alias of the loaded world model.
            config: Session configuration.
            seed: Random seed for reproducibility.

        Returns:
            A new Session instance.
        """
        from worldkernels.core.config import WorldConfig as WC
        from worldkernels.core.session import Session

        if world not in self._worlds:
            raise ValueError(f"World '{world}' not loaded. Call load_world() first.")

        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError(f"Max sessions ({self.max_sessions}) reached.")

        cfg = config or WC()
        session = Session(world_id=world, config=cfg, seed=seed or 0)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

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
