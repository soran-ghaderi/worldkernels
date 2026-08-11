r"""uv discovery in ``worldkernels.bootstrap.deps``.

uv ships as a hard dependency (the ``uv`` wheel vendors the binary), so discovery
must be PATH-independent via ``uv.find_uv_bin()``, fall back to ``shutil.which``,
and honor ``WORLDKERNELS_FORCE_PIP``. None of this may import torch.
"""

from __future__ import annotations

import sys

import pytest

from worldkernels.bootstrap import deps


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(deps, "_UV_BIN", None)
    monkeypatch.delenv("WORLDKERNELS_FORCE_PIP", raising=False)


def _never(*_a):
    pytest.fail("uv discovery was probed when it should not have been")


def test_prefers_find_uv_bin(monkeypatch):
    import uv

    monkeypatch.setattr(uv, "find_uv_bin", lambda: "/vendored/uv")
    monkeypatch.setattr(deps.shutil, "which", _never)
    assert deps._resolve_uv() == "/vendored/uv"


def test_falls_back_to_path_when_uv_unimportable(monkeypatch):
    monkeypatch.setitem(sys.modules, "uv", None)
    monkeypatch.setattr(deps.shutil, "which", lambda name: "/on/path/uv")
    assert deps._resolve_uv() == "/on/path/uv"


def test_falls_back_when_find_uv_bin_raises(monkeypatch):
    import uv

    def _missing():
        raise FileNotFoundError

    monkeypatch.setattr(uv, "find_uv_bin", _missing)
    monkeypatch.setattr(deps.shutil, "which", lambda name: "/on/path/uv")
    assert deps._resolve_uv() == "/on/path/uv"


def test_force_pip_skips_uv(monkeypatch):
    import uv

    monkeypatch.setenv("WORLDKERNELS_FORCE_PIP", "1")
    monkeypatch.setattr(uv, "find_uv_bin", _never)
    assert deps._resolve_uv() is None
    assert deps._select_installer() == "pip"


def test_select_installer_returns_uv_path(monkeypatch):
    import uv

    monkeypatch.setattr(uv, "find_uv_bin", lambda: "/vendored/uv")
    assert deps._select_installer() == "/vendored/uv"


def test_build_install_cmd_uses_uv_path():
    cmd = deps._build_install_cmd("/vendored/uv", ["pkg==1.0"], "/py", None)
    assert cmd == ["/vendored/uv", "pip", "install", "--python", "/py", "pkg==1.0"]


def test_build_install_cmd_pip_fallback():
    cmd = deps._build_install_cmd("pip", ["pkg==1.0"], "/py", None)
    assert cmd == ["/py", "-m", "pip", "install", "--disable-pip-version-check", "pkg==1.0"]
