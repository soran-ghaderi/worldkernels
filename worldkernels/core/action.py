"""Action schema for world model inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    r"""Generalized action container (control input to world)."""

    action_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if not self.action_type:
            raise ValueError("action_type cannot be empty")
