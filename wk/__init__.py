"""`wk`: short import alias for `worldkernels`. Submodules and attributes resolve
to the identical `worldkernels` objects (singletons / isinstance preserved)."""

import importlib as _importlib
import sys as _sys
from importlib.abc import Loader as _Loader, MetaPathFinder as _MetaPathFinder
from importlib.machinery import ModuleSpec as _ModuleSpec

_REAL = "worldkernels"
_ALIAS = "wk"


class _AliasLoader(_Loader):
    def __init__(self, real_name: str) -> None:
        self._real_name = real_name

    def create_module(self, spec: _ModuleSpec):
        return _importlib.import_module(self._real_name)

    def exec_module(self, module) -> None:
        pass


class _AliasFinder(_MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if fullname.startswith(_ALIAS + "."):
            real = _REAL + fullname[len(_ALIAS) :]
            return _ModuleSpec(fullname, _AliasLoader(real))
        return None


if not any(isinstance(f, _AliasFinder) for f in _sys.meta_path):
    _sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name: str):
    return getattr(_importlib.import_module(_REAL), name)
