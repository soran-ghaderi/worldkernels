r"""Tests for worldkernels/worlds/registry.py."""

from __future__ import annotations

import logging

import pytest

from tests._helpers.mocks import MockWorld
from worldkernels.worlds import registry
from worldkernels.worlds.registry import (
    _ensure_plugins_loaded,
    _register_builtins,
    get_world_class,
    list_worlds,
    register_world,
)


class TestRegisterAndLookup:
    def test_register_and_get(self):
        register_world("_pytest_w", MockWorld)
        try:
            assert get_world_class("_pytest_w") is MockWorld
        finally:
            registry._REGISTRY.pop("_pytest_w", None)

    def test_register_overwrites_with_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger="worldkernels.worlds.registry")
        register_world("_pytest_w", MockWorld)
        register_world("_pytest_w", MockWorld)
        try:
            assert any("Overwriting" in r.message for r in caplog.records)
        finally:
            registry._REGISTRY.pop("_pytest_w", None)

    def test_list_includes_builtins(self):
        names = list_worlds()
        assert "dummy" in names

    def test_list_returns_sorted(self):
        names = list_worlds()
        assert names == sorted(names)

    def test_get_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="not found in registry"):
            get_world_class("missing_world_xyz_123")


class TestHubFallback:
    def test_hub_alias_routes_to_adapter(self, monkeypatch):
        register_world("dreamdojo", MockWorld)
        try:
            cls = get_world_class("nvidia/DreamDojo")
            assert cls is MockWorld
        finally:
            from worldkernels.worlds.adapters.dreamdojo import DreamDojoWorld

            registry._REGISTRY["dreamdojo"] = DreamDojoWorld


class TestEnsurePluginsLoaded:
    def test_idempotent(self):
        registry._plugins_loaded = False
        _ensure_plugins_loaded()
        assert registry._plugins_loaded is True
        _ensure_plugins_loaded()


class TestRegisterBuiltins:
    def test_registers_dummy(self):
        registry._REGISTRY.pop("dummy", None)
        _register_builtins()
        assert "dummy" in registry._REGISTRY


class TestRegisterBuiltinsImportErrors:
    def test_dreamdojo_failure_logged(self, monkeypatch, caplog):
        import builtins
        import logging

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "worldkernels.worlds.adapters.dreamdojo":
                raise ImportError("missing dep")
            return real_import(name, *args, **kwargs)

        caplog.set_level(logging.DEBUG, logger="worldkernels.worlds.registry")
        with monkeypatch.context() as m:
            m.setattr(builtins, "__import__", fake_import)
            registry._REGISTRY.pop("dreamdojo", None)
            _register_builtins()

    def test_cosmos_failure_logged(self, monkeypatch, caplog):
        import builtins
        import logging

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "worldkernels.worlds.adapters.cosmos":
                raise ImportError("missing dep")
            return real_import(name, *args, **kwargs)

        caplog.set_level(logging.DEBUG, logger="worldkernels.worlds.registry")
        with monkeypatch.context() as m:
            m.setattr(builtins, "__import__", fake_import)
            registry._REGISTRY.pop("cosmos_predict2", None)
            _register_builtins()


class TestPluginsDiscovery:
    def _reset_plugins(self):
        registry._plugins_loaded = False

    def test_select_path_loads_plugin(self, monkeypatch):
        from unittest.mock import MagicMock

        self._reset_plugins()

        class _FakeEP:
            def __init__(self, name, cls):
                self.name = name
                self._cls = cls

            def load(self):
                return self._cls

        class _PlugWorld(MockWorld):
            name = "_plug"

        eps = MagicMock()
        eps.select.return_value = [_FakeEP("_plug_pytest", _PlugWorld)]
        monkeypatch.setattr("importlib.metadata.entry_points", lambda: eps)
        registry._ensure_plugins_loaded()
        try:
            assert "_plug_pytest" in registry._REGISTRY
        finally:
            registry._REGISTRY.pop("_plug_pytest", None)

    def test_plugin_load_failure_is_logged(self, monkeypatch, caplog):
        import logging
        from unittest.mock import MagicMock

        self._reset_plugins()

        class _FailingEP:
            name = "_fail"

            def load(self):
                raise RuntimeError("boom")

        eps = MagicMock()
        eps.select.return_value = [_FailingEP()]
        monkeypatch.setattr("importlib.metadata.entry_points", lambda: eps)
        caplog.set_level(logging.WARNING, logger="worldkernels.worlds.registry")
        registry._ensure_plugins_loaded()
        assert any("Failed to load world plugin" in r.message for r in caplog.records)

    def test_entry_points_dict_fallback(self, monkeypatch):
        self._reset_plugins()

        class _OldStyleEPs(dict):
            pass

        eps = _OldStyleEPs()
        monkeypatch.setattr("importlib.metadata.entry_points", lambda: eps)
        registry._ensure_plugins_loaded()

    def test_entry_points_failure_is_logged(self, monkeypatch):
        self._reset_plugins()
        monkeypatch.setattr(
            "importlib.metadata.entry_points",
            lambda: (_ for _ in ()).throw(RuntimeError("broken")),
        )
        registry._ensure_plugins_loaded()

    def test_skip_overwrite_of_existing(self, monkeypatch):
        from unittest.mock import MagicMock

        self._reset_plugins()

        class _FakeEP:
            def __init__(self, name, cls):
                self.name = name
                self._cls = cls

            def load(self):
                return self._cls

        eps = MagicMock()
        eps.select.return_value = [_FakeEP("dummy", MockWorld)]
        monkeypatch.setattr("importlib.metadata.entry_points", lambda: eps)
        from worldkernels.worlds.adapters.dummy import DummyWorld

        registry._REGISTRY["dummy"] = DummyWorld
        registry._ensure_plugins_loaded()
        assert registry._REGISTRY["dummy"] is DummyWorld
