r"""Unified phase-event model shared by the progress UI and SSE streaming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PhaseStatus = Literal["pending", "running", "done", "skipped", "failed"]


@dataclass(frozen=True)
class PhaseEvent:
    phase: str
    status: PhaseStatus
    message: str = ""
    fraction: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "message": self.message,
            "fraction": self.fraction,
        }
