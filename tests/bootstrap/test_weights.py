r"""Tests for worldkernels.bootstrap.weights.provision_weights.

Regression guard: a model whose card declares a ``weights_provider`` must NOT
fall through to an unscoped ``snapshot_download`` (which for the DreamDojo
monorepo mirrors ~859 GB instead of the one variant's ~13 GB of model shards).
"""

from __future__ import annotations

import pytest

from worldkernels.bootstrap import weights as _weights
from worldkernels.bootstrap.weights import provision_weights as _real_provision
from worldkernels.worlds.hub import ModelCard, get_model_card


@pytest.fixture(autouse=True)
def _allow_fetch_env(monkeypatch):
    monkeypatch.delenv("WORLDKERNELS_NO_AUTO_INSTALL", raising=False)


def _spy_snapshot(monkeypatch):
    calls = []

    def snap(repo, allow_patterns=None, repo_type="model"):
        calls.append({"repo": repo, "allow_patterns": allow_patterns})
        return "/fake/snapshot"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snap)
    return calls


def test_provider_card_skips_unscoped_snapshot(monkeypatch):
    seen = {}

    def fake_provider(variant=None):
        seen["variant"] = variant
        return "/converted/model_ema_bf16.pt"

    monkeypatch.setattr(
        _weights, "_load_weights_provider", lambda spec: fake_provider, raising=False
    )
    snap_calls = _spy_snapshot(monkeypatch)

    card = ModelCard(
        adapter="dreamdojo",
        hf_repo="nvidia/DreamDojo",
        weights_provider="x:y",
    )
    out = _real_provision(card, variant="2b_pretrain", allow_fetch=True)

    assert out == "/converted/model_ema_bf16.pt"
    assert seen["variant"] == "2b_pretrain"
    assert snap_calls == [], "provider card must not trigger a whole-repo snapshot_download"


def test_dreamdojo_card_has_no_unscoped_download():
    card = get_model_card("dreamdojo")
    assert card is not None
    assert card.weights_provider is not None, (
        "dreamdojo card must declare a weights_provider so provision_weights does not "
        "mirror the entire 859 GB monorepo"
    )


def test_generic_card_still_uses_snapshot(monkeypatch):
    snap_calls = _spy_snapshot(monkeypatch)
    card = ModelCard(adapter="generator_world", hf_repo="some/repo")
    out = _real_provision(card, allow_fetch=True)
    assert out == "/fake/snapshot"
    assert len(snap_calls) == 1
    assert snap_calls[0]["repo"] == "some/repo"
