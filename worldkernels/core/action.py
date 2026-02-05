"""Action schema for world model inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """
    Generalized action container.

    Not RL-specific — just "control input to world".

    Examples:
        >>> Action("keyboard", {"keys": ["W", "SPACE"]})
        >>> Action("continuous", {"velocity": [0.1, 0.0, 0.5]})
        >>> Action("camera", {"trajectory": "orbit_left_30"})
        >>> Action("text", {"command": "open the door"})
    """

    action_type: str
    """Type of action, e.g., 'keyboard', 'continuous', 'text', 'pose'."""

    payload: dict[str, Any] = field(default_factory=dict)
    """Schema depends on action_type."""

    timestamp: float | None = None
    """Optional timing information."""

    def __post_init__(self) -> None:
        if not self.action_type:
            raise ValueError("action_type cannot be empty")
