r"""Tests for worldkernels.bootstrap.hf transfer-backend tuning.

Regression guard: Xet high-performance mode must stay OFF on unauthenticated
transfers, where its aggressive concurrency triggers hub rate-limiting and can
deadlock ``hf_xet`` (sockets stuck in CLOSE-WAIT, download stalls).
"""

from __future__ import annotations

import os

import pytest

from worldkernels.bootstrap import hf


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "HF_XET_HIGH_PERFORMANCE",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def _force_xet_present(monkeypatch):
    monkeypatch.setattr(hf._util, "find_spec", lambda name: object() if name == "hf_xet" else None)


def test_high_performance_disabled_without_token(monkeypatch):
    _force_xet_present(monkeypatch)
    monkeypatch.setattr(hf, "_has_hf_token", lambda: False)
    hf.enable_fast_hf_transfer()
    assert "HF_XET_HIGH_PERFORMANCE" not in os.environ


def test_high_performance_enabled_with_token(monkeypatch):
    _force_xet_present(monkeypatch)
    monkeypatch.setattr(hf, "_has_hf_token", lambda: True)
    hf.enable_fast_hf_transfer()
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"


def test_has_hf_token_reads_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    assert hf._has_hf_token() is True


def test_has_hf_token_false_when_absent(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None, raising=False)
    assert hf._has_hf_token() is False
