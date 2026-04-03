r"""World model interface, registry, and adapters."""

from worldkernels.worlds.base import AbstractWorld
from worldkernels.worlds.registry import get_world_class, list_worlds, register_world
__all__ = ["AbstractWorld", "get_world_class", "list_worlds", "register_world"]
