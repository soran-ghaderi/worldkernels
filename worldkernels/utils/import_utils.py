r"""Deferred optional-dependency handling (ADR-013).

A missing optional package becomes a `PlaceholderModule` rather than an
``ImportError`` at import time. The placeholder is safe to pass around and
assign; it raises only when an attribute is *used*, with a message naming the
``worldkernels`` extra to install. Mirrors vLLM's ``vllm/utils/import_utils.py``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

__all__ = ["PlaceholderModule", "optional_import", "LazyLoader"]


class _PlaceholderBase:
    def _raise(self) -> Any:
        raise NotImplementedError


class _PlaceholderModuleAttr(_PlaceholderBase):
    def __init__(self, module: "PlaceholderModule", path: str) -> None:
        self._module = module
        self._path = path

    def __getattr__(self, name: str) -> "_PlaceholderModuleAttr":
        return _PlaceholderModuleAttr(self._module, f"{self._path}.{name}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._module._raise(self._path)


class PlaceholderModule(_PlaceholderBase):
    r"""Stand-in for a missing optional package.

    Accessing any attribute raises `ImportError` naming the extra to
    install. Construction never raises, so a placeholder can be bound and
    passed around exactly like the real module.

    Args:
        pkg_name: The import name that failed (e.g. ``"flash_attn"``).
        extra: The ``worldkernels`` extra that provides it (e.g. ``"flash-attn"``).
    """

    def __init__(self, pkg_name: str, extra: str) -> None:
        self._pkg_name = pkg_name
        self._extra = extra

    def _raise(self, path: str | None = None) -> Any:
        target = f"{self._pkg_name}.{path}" if path else self._pkg_name
        raise ImportError(
            f"{target!r} requires the optional package {self._pkg_name!r}. "
            f"Install it with: pip install 'worldkernels[{self._extra}]'"
        )

    def __getattr__(self, name: str) -> _PlaceholderModuleAttr:
        return _PlaceholderModuleAttr(self, name)


def optional_import(pkg: str, extra: str) -> ModuleType | PlaceholderModule:
    r"""Import ``pkg``, returning a `PlaceholderModule` if it is absent.

    Args:
        pkg: Import name of the optional package.
        extra: The ``worldkernels`` extra that provides it.
    """
    try:
        return importlib.import_module(pkg)
    except ImportError:
        return PlaceholderModule(pkg, extra)


class LazyLoader(ModuleType):
    r"""Defer a module's import to first attribute access.

    Useful for modules with import-time side effects (e.g. op registration) that
    we want to keep out of the cold path. On first attribute access, the real
    module is imported and patched into ``parent_globals`` + ``sys.modules`` so
    subsequent accesses are free.

    Args:
        local_name: Binding name in the caller's module (e.g. ``"diffusers"``).
        parent_globals: The caller's ``globals()`` dict.
        module_name: Fully qualified import path.
    """

    def __init__(self, local_name: str, parent_globals: dict, module_name: str) -> None:
        super().__init__(module_name)
        self._local_name = local_name
        self._parent_globals = parent_globals
        self._module_name = module_name

    def _load(self) -> ModuleType:
        import sys as _sys

        module = importlib.import_module(self._module_name)
        self._parent_globals[self._local_name] = module
        _sys.modules[self._module_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: str) -> Any:
        module = self._load()
        return getattr(module, item)

    def __dir__(self) -> list[str]:
        module = self._load()
        return dir(module)
