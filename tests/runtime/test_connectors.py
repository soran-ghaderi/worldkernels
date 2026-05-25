r"""Tests for worldkernels/runtime/connectors.py."""

from __future__ import annotations

import pytest

from worldkernels.runtime.connectors import (
    ConnectorRegistry,
    LocalConnector,
    StageConnector,
)
from worldkernels.runtime.stages import StageOutput, StageType


def _output() -> StageOutput:
    return StageOutput(stage_type=StageType.ENCODE, data=[1, 2, 3])


class TestLocalConnector:
    def test_put_then_get_returns_and_consumes(self):
        c = LocalConnector()
        out = _output()
        assert c.put("encode", "transition", "k1", out) is True
        got = c.get("encode", "transition", "k1")
        assert got is out
        assert c.get("encode", "transition", "k1") is None

    def test_get_missing_returns_none(self):
        c = LocalConnector()
        assert c.get("a", "b", "k") is None

    def test_cleanup_drops_matching_keys_only(self):
        c = LocalConnector()
        c.put("a", "b", "k1", _output())
        c.put("a", "b", "k2", _output())
        c.put("a", "c", "k1", _output())
        c.cleanup("k1")
        assert c.get("a", "b", "k1") is None
        assert c.get("a", "c", "k1") is None
        assert c.get("a", "b", "k2") is not None


class TestStageConnectorABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            StageConnector()  # type: ignore[abstract]


class TestConnectorRegistry:
    def teardown_method(self):
        ConnectorRegistry._registry = {"local": LocalConnector}

    def test_local_is_preregistered(self):
        assert "local" in ConnectorRegistry.list_registered()

    def test_register_and_create(self):
        class Dummy(LocalConnector):
            pass

        ConnectorRegistry.register("dummy", Dummy)
        instance = ConnectorRegistry.create("dummy")
        assert isinstance(instance, Dummy)

    def test_register_overwrite_logs_warning(self, caplog):
        import logging

        caplog.set_level(logging.WARNING, logger="worldkernels.runtime.connectors")
        ConnectorRegistry.register("local", LocalConnector)
        assert any("Overwriting" in r.message for r in caplog.records)

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown connector"):
            ConnectorRegistry.create("missing")
