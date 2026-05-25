r"""Tests for worldkernels/core/action.py."""

from __future__ import annotations

import pytest

from worldkernels.core.action import Action


class TestActionConstruction:
    def test_minimal(self):
        a = Action(action_type="null")
        assert a.action_type == "null"
        assert a.payload == {}
        assert a.timestamp is None

    def test_full(self):
        a = Action(action_type="keyboard", payload={"keys": ["W"]}, timestamp=1.5)
        assert a.action_type == "keyboard"
        assert a.payload == {"keys": ["W"]}
        assert a.timestamp == 1.5

    def test_default_payload_is_independent_per_instance(self):
        a, b = Action("x"), Action("y")
        a.payload["k"] = 1
        assert "k" not in b.payload

    def test_empty_action_type_rejected(self):
        with pytest.raises(ValueError, match="action_type"):
            Action(action_type="")
