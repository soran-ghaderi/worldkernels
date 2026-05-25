"""World model registry with built-in adapters and entry_points plugin discovery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worldkernels.worlds.base import WorldModel

log = logging.getLogger(__name__)

# name -> WorldModel subclass (not instances, classes)
_REGISTRY: dict[str, type[WorldModel]] = {}
_plugins_loaded = False


def register_world(name: str, cls: type[WorldModel]) -> None:
    if name in _REGISTRY:
        log.warning("Overwriting existing world registration: %s", name)
    _REGISTRY[name] = cls
    log.debug("Registered world: %s -> %s", name, cls.__qualname__)


def get_world_class(name: str) -> type[WorldModel]:
    r"""Look up a registered world model class by name.

    Checks the adapter registry first, then resolves through the hub
    (which maps HF repo IDs and short aliases to adapter names).
    """
    _ensure_plugins_loaded()
    if name in _REGISTRY:
        return _REGISTRY[name]

    from worldkernels.worlds.hub import get_model_card

    card = get_model_card(name)
    if card is not None and card.adapter in _REGISTRY:
        return _REGISTRY[card.adapter]

    available = ", ".join(sorted(_REGISTRY)) or "(none)"
    raise KeyError(f"World '{name}' not found in registry. Available: {available}")


def list_worlds() -> list[str]:
    r"""Return sorted list of all registered world names."""
    _ensure_plugins_loaded()
    return sorted(_REGISTRY)


# ---- built-in registrations ---------------------------------------------


def _register_builtins() -> None:
    from worldkernels.worlds.base import GeneratorWorld
    from worldkernels.worlds.dummy import DummyWorld

    register_world("dummy", DummyWorld)
    register_world("generator_world", GeneratorWorld)

    try:
        from worldkernels.worlds.dreamdojo import DreamDojoWorld

        register_world("dreamdojo", DreamDojoWorld)
    except ImportError:
        log.debug("DreamDojo world not available (cosmos_predict2 not installed)")


# ---- entry_points discovery (lazy, once) --------------------------------


def _ensure_plugins_loaded() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    # Built-ins first
    _register_builtins()

    # Then third-party plugins via entry_points
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        # Python 3.12+ returns a SelectableGroups; 3.9-3.11 returns dict
        if hasattr(eps, "select"):
            world_eps = eps.select(group="worldkernels.worlds")
        else:
            world_eps = eps.get("worldkernels.worlds", [])  # type: ignore[arg-type]

        for ep in world_eps:
            try:
                cls = ep.load()
                # Don't overwrite built-ins with their own entry_points
                if ep.name not in _REGISTRY:
                    register_world(ep.name, cls)
            except Exception:
                log.warning("Failed to load world plugin: %s", ep.name, exc_info=True)
    except Exception:
        log.debug("entry_points discovery unavailable", exc_info=True)
