r"""Tests for worldkernels.models.cosmos_predict2.checkpoint (scoped download)."""

from __future__ import annotations

import pytest

from worldkernels.models.cosmos_predict2 import checkpoint as ckpt


def test_resolve_ckpt_file_default():
    assert ckpt.resolve_ckpt_file(None) == ckpt.COSMOS_CKPT_FILES["pretrained"]
    assert ckpt.resolve_ckpt_file("unknown") == ckpt.COSMOS_CKPT_FILES["pretrained"]


def test_resolve_ckpt_file_variant():
    assert ckpt.resolve_ckpt_file("distilled") == ckpt.COSMOS_CKPT_FILES["distilled"]
    assert "post-trained" in ckpt.COSMOS_CKPT_FILES


def test_provision_fetches_only_ckpt_and_tokenizer(monkeypatch):
    calls = []

    def fake_dl(repo, filename=None, repo_type="model"):
        calls.append((repo, filename))
        return f"/cache/{filename}"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("must not snapshot")
    )

    out = ckpt.provision_cosmos_weights("distilled")

    assert out == f"/cache/{ckpt.COSMOS_CKPT_FILES['distilled']}"
    assert (ckpt.COSMOS_HF_REPO, ckpt.COSMOS_CKPT_FILES["distilled"]) in calls
    assert (ckpt.WAN_VAE_REPO, ckpt.WAN_VAE_FILE) in calls
    assert len(calls) == 2


def test_vae_tokenizer_uses_non_gated_wan_repo(monkeypatch):
    r"""The VAE must come from the public Wan-AI repo, not the gated Cosmos mirror."""
    calls = []

    def fake_dl(repo, filename=None, repo_type="model"):
        calls.append((repo, filename))
        return f"/cache/{filename}"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    out = ckpt.download_vae_tokenizer()

    assert out == f"/cache/{ckpt.WAN_VAE_FILE}"
    assert calls == [(ckpt.WAN_VAE_REPO, ckpt.WAN_VAE_FILE)]
    assert ckpt.WAN_VAE_REPO != ckpt.COSMOS_HF_REPO
