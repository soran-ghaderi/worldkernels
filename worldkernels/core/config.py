r"""Configuration dataclasses (moved to `config`).

This module re-exports during the architecture restructure; importers are
migrated to `config` and this file is then removed.
"""

from __future__ import annotations

from worldkernels.config import ServerConfig, WorldConfig

__all__ = ["WorldConfig", "ServerConfig"]
