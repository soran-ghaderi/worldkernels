"""
WorldKernels - GPU-first world model simulation engine.

Serve learned world models (DiT, VAE) as interactive sessions.

Example:
    >>> from worldkernels import WorldKernel, Action, WorldConfig
    >>> wk = WorldKernel(device="cuda")
    >>> wk.load_world("Etched/oasis-500m")
    >>> session = wk.create_session(world="oasis-500m", config=WorldConfig())
    >>> obs = session.step(Action("keyboard", {"keys": ["W"]}))
"""

from worldkernels.core.action import Action
from worldkernels.core.config import ServerConfig, WorldConfig
from worldkernels.core.engine import WorldKernel
from worldkernels.core.observation import Observation
from worldkernels.core.session import Session, SessionStatus

try:
    from worldkernels._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.1.0.dev0"
    __version_tuple__ = (0, 1, 0, "dev0")

__all__ = [
    "WorldKernel",
    "Session",
    "SessionStatus",
    "Action",
    "Observation",
    "WorldConfig",
    "ServerConfig",
    "__version__",
]
