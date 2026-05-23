r"""HTTP/WebSocket server configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ServerConfig"]


@dataclass
class ServerConfig:
    r"""Configuration for the serving layer."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_sessions: int = 4
    api_key: str | None = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
