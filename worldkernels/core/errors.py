"""Exception hierarchy for WorldKernels."""

from __future__ import annotations


class WorldKernelError(Exception):
    pass


class WorldNotFoundError(WorldKernelError):
    def __init__(self, world_id: str) -> None:
        self.world_id = world_id
        super().__init__(f"World '{world_id}' not loaded. Call load_world() first.")


class SessionLimitError(WorldKernelError):
    def __init__(self, max_sessions: int) -> None:
        self.max_sessions = max_sessions
        super().__init__(f"Max sessions ({max_sessions}) reached.")


class SessionNotFoundError(WorldKernelError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' not found.")


class SessionTerminatedError(WorldKernelError):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' is terminated.")


class VRAMExhaustedError(WorldKernelError):
    def __init__(self, required_mb: float, available_mb: float) -> None:
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            f"Not enough VRAM: need {required_mb:.0f} MB, "
            f"only {available_mb:.0f} MB available."
        )


class WorldInitError(WorldKernelError):
    def __init__(self, world_id: str, reason: str) -> None:
        self.world_id = world_id
        self.reason = reason
        super().__init__(f"Failed to initialize world '{world_id}': {reason}")
