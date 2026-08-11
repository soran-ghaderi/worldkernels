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


def load_action_chunks(path: str, steps: int, chunk_size: int = 12) -> list[Any]:
    r"""Load an action ``.npy`` into per-step chunks of ``[chunk_size, action_dim]``.

    Accepts ``[T, action_dim]`` (split into ``chunk_size``-frame chunks) or pre-chunked
    ``[num_chunks, chunk_size, action_dim]``. DreamDojo GR-1 expects ``action_dim=384``.
    """
    from pathlib import Path

    import numpy as np

    arr = np.load(Path(path).expanduser())
    if arr.ndim == 3:
        chunks = [arr[i] for i in range(arr.shape[0])]
    elif arr.ndim == 2:
        chunks = [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]
    else:
        raise ValueError(f"actions must be 2D [T,D] or 3D [N,chunk,D], got shape {arr.shape}")
    return chunks[:steps] if steps < len(chunks) else chunks
