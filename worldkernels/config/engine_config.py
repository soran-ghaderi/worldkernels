r"""Engine configuration.

``EngineConfig`` is retained as a deprecated alias for `RuntimeConfig`, which
is now the single source of truth for engine settings + component toggles.
"""

from __future__ import annotations

from worldkernels.config.runtime import RuntimeConfig

__all__ = ["EngineConfig"]

EngineConfig = RuntimeConfig
