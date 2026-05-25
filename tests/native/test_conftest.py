r"""Tests for tests/native/conftest.py (CLI-flag parsing + fixture wiring)."""

from __future__ import annotations


def test_fixture_cfg_uses_defaults(pytestconfig):
    from tests.native.conftest import _parse_spatial

    assert _parse_spatial("240x320") == (240, 320)
    assert _parse_spatial("64X64") == (64, 64)
