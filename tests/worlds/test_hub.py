r"""Tests for worldkernels/worlds/hub.py."""

from __future__ import annotations

import sys

import pytest

from worldkernels.worlds import hub
from worldkernels.worlds.hub import (
    ModelCard,
    ensure_model_deps,
    get_model_card,
    list_models,
    register_model,
    resolve_model,
)


class TestModelCard:
    def test_defaults(self):
        c = ModelCard(adapter="foo")
        assert c.adapter == "foo"
        assert c.hf_repo is None
        assert c.default_kwargs == {}
        assert c.description == ""
        assert c.pip_extra is None

    def test_full(self):
        c = ModelCard(
            adapter="foo",
            hf_repo="owner/repo",
            default_kwargs={"variant": "x"},
            description="desc",
            pip_extra="cosmos",
        )
        assert c.adapter == "foo"
        assert c.hf_repo == "owner/repo"
        assert c.default_kwargs == {"variant": "x"}
        assert c.pip_extra == "cosmos"

    def test_frozen(self):
        c = ModelCard(adapter="foo")
        with pytest.raises(AttributeError):
            c.adapter = "bar"


class TestRegistry:
    def test_register_and_lookup(self):
        register_model("_pytest_card", ModelCard(adapter="dummy"))
        try:
            assert get_model_card("_pytest_card").adapter == "dummy"
        finally:
            hub._HUB.pop("_pytest_card", None)

    def test_get_unknown_returns_none(self):
        assert get_model_card("not_a_card_xyz") is None

    def test_list_models_includes_builtins(self):
        models = list_models()
        assert "dummy" in models
        assert "dreamdojo" in models
        assert "cosmos_predict2" in models
        assert "nvidia/DreamDojo" in models

    def test_list_models_returns_copy(self):
        a = list_models()
        b = list_models()
        a["__x__"] = ModelCard(adapter="x")
        assert "__x__" not in b


class TestResolveModel:
    def test_known_alias(self):
        adapter, kw = resolve_model("dreamdojo-2b-pretrain")
        assert adapter == "dreamdojo"
        assert kw == {"variant": "2b_pretrain"}

    def test_user_kwargs_override_defaults(self):
        adapter, kw = resolve_model("dreamdojo-2b-pretrain", variant="custom")
        assert kw == {"variant": "custom"}

    def test_user_kwargs_merge_with_defaults(self):
        adapter, kw = resolve_model("dreamdojo-2b-pretrain", extra=42)
        assert kw == {"variant": "2b_pretrain", "extra": 42}

    def test_unknown_falls_through(self):
        adapter, kw = resolve_model("custom_model_xyz", foo=1)
        assert adapter == "custom_model_xyz"
        assert kw == {"foo": 1}

    def test_hf_repo_id_resolves(self):
        adapter, kw = resolve_model("nvidia/DreamDojo")
        assert adapter == "dreamdojo"
        assert kw == {"variant": "2b_pretrain"}


class TestEnsureModelDeps:
    def test_unknown_model_is_noop(self, monkeypatch):
        called = False

        def fake_install(*args, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr("subprocess.check_call", fake_install)
        ensure_model_deps("not_a_known_model")
        assert called is False

    def test_card_without_pip_extra_is_noop(self, monkeypatch):
        monkeypatch.setattr("subprocess.check_call", lambda *a, **kw: setattr(self_, "_c", True))
        register_model("_no_extra", ModelCard(adapter="dummy"))
        try:
            ensure_model_deps("_no_extra")
        finally:
            hub._HUB.pop("_no_extra", None)

    def test_unknown_sentinel_is_noop(self, monkeypatch):
        called = []
        monkeypatch.setattr("subprocess.check_call", lambda *a, **kw: called.append(1))
        register_model("_weird", ModelCard(adapter="dummy", pip_extra="unknown_extra"))
        try:
            ensure_model_deps("_weird")
        finally:
            hub._HUB.pop("_weird", None)
        assert called == []

    def test_sentinel_present_skips_install(self, monkeypatch):
        called = []
        monkeypatch.setattr("subprocess.check_call", lambda *a, **kw: called.append(1))
        monkeypatch.setitem(hub._EXTRA_SENTINELS, "cosmos", "json")
        ensure_model_deps("cosmos_predict2")
        assert called == []

    def test_sentinel_missing_triggers_install(self, monkeypatch):
        called = []
        monkeypatch.setitem(hub._EXTRA_SENTINELS, "cosmos", "definitely_not_installed_xyz")
        monkeypatch.delenv("WORLDKERNELS_NO_AUTO_INSTALL", raising=False)
        monkeypatch.setattr("subprocess.check_call", lambda *a, **kw: called.append(a[0]))
        ensure_model_deps("cosmos_predict2")
        assert called and "pip" in called[0]

    def test_sentinel_missing_with_no_auto_install_raises(self, monkeypatch):
        monkeypatch.setitem(hub._EXTRA_SENTINELS, "cosmos", "definitely_not_installed_xyz")
        monkeypatch.setenv("WORLDKERNELS_NO_AUTO_INSTALL", "1")
        with pytest.raises(ImportError, match="Missing dependencies"):
            ensure_model_deps("cosmos_predict2")

    def test_sentinel_already_in_sys_modules(self, monkeypatch):
        called = []
        monkeypatch.setitem(hub._EXTRA_SENTINELS, "cosmos", "json")
        monkeypatch.setattr("subprocess.check_call", lambda *a, **kw: called.append(1))
        assert "json" in sys.modules
        ensure_model_deps("cosmos_predict2")
        assert called == []


class TestBuiltinRegistrations:
    @pytest.mark.parametrize(
        "alias",
        [
            "dummy",
            "dreamdojo",
            "dreamdojo-2b-pretrain",
            "dreamdojo-2b-gr1",
            "dreamdojo-2b-agibot",
            "dreamdojo-14b-pretrain",
            "cosmos-predict2",
            "cosmos_predict2",
            "nvidia/Cosmos-Predict2.5-2B",
            "nvidia/DreamDojo",
        ],
    )
    def test_alias_registered(self, alias):
        card = get_model_card(alias)
        assert card is not None
        assert card.adapter in {"dummy", "dreamdojo", "cosmos_predict2"}


self_ = type("_X", (), {})()
