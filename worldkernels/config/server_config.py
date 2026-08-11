r"""HTTP/WebSocket server configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["ServerConfig"]


@dataclass
class ServerConfig:
    r"""Configuration for the serving layer."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_sessions: int = 4
    api_key: str | None = None
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    allowed_methods: list[str] = field(default_factory=lambda: ["*"])
    allowed_headers: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False
    ssl_keyfile: str | None = None
    ssl_certfile: str | None = None
    uvicorn_log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = "info"
    disable_access_log: bool = False
    root_path: str = ""
