r"""World-model taxonomy, registry, and hub."""

import importlib as _importlib

__all__ = [
    "WorldModel",
    "InteractiveWorldModel",
    "GeneratorWorld",
    "AbstractWorld",
    "get_world_class",
    "list_worlds",
    "register_world",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "WorldModel": ("worldkernels.worlds.base", "WorldModel"),
    "InteractiveWorldModel": ("worldkernels.worlds.base", "InteractiveWorldModel"),
    "GeneratorWorld": ("worldkernels.worlds.base", "GeneratorWorld"),
    "AbstractWorld": ("worldkernels.worlds.base", "AbstractWorld"),
    "get_world_class": ("worldkernels.worlds.registry", "get_world_class"),
    "list_worlds": ("worldkernels.worlds.registry", "list_worlds"),
    "register_world": ("worldkernels.worlds.registry", "register_world"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'worldkernels.worlds' has no attribute {name!r}")
