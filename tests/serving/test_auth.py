r"""Tests for worldkernels/serving/auth.py."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from worldkernels.serving.auth import require_api_key


def _call(dep, key):
    return asyncio.run(dep(key=key))


class TestRequireApiKey:
    def test_no_expected_key_allows_any(self):
        dep = require_api_key(None)
        _call(dep, None)
        _call(dep, "anything")
        _call(dep, "Bearer abc")

    def test_missing_key_raises_401(self):
        dep = require_api_key("expected")
        with pytest.raises(HTTPException) as ei:
            _call(dep, None)
        assert ei.value.status_code == 401
        assert "Missing" in ei.value.detail

    def test_bearer_prefix_stripped(self):
        dep = require_api_key("expected")
        _call(dep, "Bearer expected")

    def test_raw_token_accepted(self):
        dep = require_api_key("expected")
        _call(dep, "expected")

    def test_wrong_key_raises_401(self):
        dep = require_api_key("expected")
        with pytest.raises(HTTPException) as ei:
            _call(dep, "Bearer wrong")
        assert ei.value.status_code == 401
        assert "Invalid" in ei.value.detail
