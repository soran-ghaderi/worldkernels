"""WorldKernels - GPU-first world model simulation engine."""

import importlib as _importlib

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

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Action": ("worldkernels.core.action", "Action"),
    "WorldConfig": ("worldkernels.core.config", "WorldConfig"),
    "ServerConfig": ("worldkernels.core.config", "ServerConfig"),
    "WorldKernel": ("worldkernels.core.engine", "WorldKernel"),
    "Observation": ("worldkernels.core.observation", "Observation"),
    "Session": ("worldkernels.core.session", "Session"),
    "SessionStatus": ("worldkernels.core.session", "SessionStatus"),
    "LatentState": ("worldkernels.core.session", "LatentState"),
    "StageConfig": ("worldkernels.runtime.stages", "StageConfig"),
    "StageExecMode": ("worldkernels.runtime.stages", "StageExecMode"),
    "StageOutput": ("worldkernels.runtime.stages", "StageOutput"),
    "StageTiming": ("worldkernels.runtime.stages", "StageTiming"),
    "StageType": ("worldkernels.runtime.stages", "StageType"),
    "TransitionMode": ("worldkernels.runtime.stages", "TransitionMode"),
    "WorldKernelError": ("worldkernels.core.errors", "WorldKernelError"),
    "WorldNotFoundError": ("worldkernels.core.errors", "WorldNotFoundError"),
    "WorldAlreadyLoadedError": ("worldkernels.core.errors", "WorldAlreadyLoadedError"),
    "WorldInitError": ("worldkernels.core.errors", "WorldInitError"),
    "SessionLimitError": ("worldkernels.core.errors", "SessionLimitError"),
    "SessionNotFoundError": ("worldkernels.core.errors", "SessionNotFoundError"),
    "SessionTerminatedError": ("worldkernels.core.errors", "SessionTerminatedError"),
    "SessionPausedError": ("worldkernels.core.errors", "SessionPausedError"),
    "CheckpointNotFoundError": ("worldkernels.core.errors", "CheckpointNotFoundError"),
    "VRAMExhaustedError": ("worldkernels.core.errors", "VRAMExhaustedError"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'worldkernels' has no attribute {name!r}")
