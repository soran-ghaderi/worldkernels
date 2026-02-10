"""WorldKernels - GPU-first world model simulation engine."""

from worldkernels.core.action import Action
from worldkernels.core.config import ServerConfig, WorldConfig
from worldkernels.core.engine import WorldKernel
from worldkernels.core.errors import (
    CheckpointNotFoundError,
    SessionLimitError,
    SessionNotFoundError,
    SessionPausedError,
    SessionTerminatedError,
    VRAMExhaustedError,
    WorldAlreadyLoadedError,
    WorldInitError,
    WorldKernelError,
    WorldNotFoundError,
)
from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState, Session, SessionStatus
from worldkernels.runtime.stages import (
    StageConfig,
    StageExecMode,
    StageOutput,
    StageTiming,
    StageType,
    TransitionMode,
)

try:
    from worldkernels._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.1.0.dev0"
    __version_tuple__ = (0, 1, 0, "dev0")

__all__ = [
    "WorldKernel",
    "Session",
    "SessionStatus",
    "LatentState",
    "Action",
    "Observation",
    "WorldConfig",
    "ServerConfig",
    "StageConfig",
    "StageExecMode",
    "StageOutput",
    "StageTiming",
    "StageType",
    "TransitionMode",
    "WorldKernelError",
    "WorldNotFoundError",
    "WorldAlreadyLoadedError",
    "WorldInitError",
    "SessionLimitError",
    "SessionNotFoundError",
    "SessionTerminatedError",
    "SessionPausedError",
    "CheckpointNotFoundError",
    "VRAMExhaustedError",
    "__version__",
]
