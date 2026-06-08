r"""Tests for worldkernels.models.dreamdojo.checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldkernels.models.dreamdojo import checkpoint as ckpt


def test_provision_maps_variant_to_ckpt_dir(monkeypatch):
    seen = {}

    def fake_download(d):
        seen["dir"] = d
        return "/p.pt"

    monkeypatch.setattr(ckpt, "download_dreamdojo_checkpoint", fake_download)
    assert ckpt.provision_dreamdojo_weights("2b_gr1") == "/p.pt"
    assert seen["dir"] == ckpt.CKPT_DIRS["2b_gr1"]


def test_provision_defaults_unknown_variant(monkeypatch):
    seen = {}

    def fake_download(d):
        seen["dir"] = d
        return "/p.pt"

    monkeypatch.setattr(ckpt, "download_dreamdojo_checkpoint", fake_download)
    ckpt.provision_dreamdojo_weights("nonsense")
    assert seen["dir"] == "2B_pretrain"


def test_provision_none_variant_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(ckpt, "download_dreamdojo_checkpoint", lambda d: seen.__setitem__("dir", d))
    ckpt.provision_dreamdojo_weights(None)
    assert seen["dir"] == "2B_pretrain"


def _patch_latest(monkeypatch, tmp_path, content):
    import huggingface_hub

    f = tmp_path / "latest_checkpoint.txt"
    f.write_text(content)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(f))


def test_invalid_iter_name_raises(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setenv("WORLDKERNELS_HOME", str(tmp_path / "wk"))
    _patch_latest(monkeypatch, tmp_path, "   ")
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("must not download")
    )
    with pytest.raises(RuntimeError, match="invalid latest iter"):
        ckpt.download_dreamdojo_checkpoint("2B_pretrain")


def test_iter_name_with_slash_raises(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setenv("WORLDKERNELS_HOME", str(tmp_path / "wk"))
    _patch_latest(monkeypatch, tmp_path, "iter_1/../../etc")
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("must not download")
    )
    with pytest.raises(RuntimeError, match="invalid latest iter"):
        ckpt.download_dreamdojo_checkpoint("2B_pretrain")


def test_missing_model_shards_raises(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setenv("WORLDKERNELS_HOME", str(tmp_path / "wk"))
    _patch_latest(monkeypatch, tmp_path, "iter_000140000")

    def fake_snap(*a, local_dir=None, **k):
        Path(local_dir, "2B_pretrain", "iter_000140000", "model").mkdir(parents=True)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snap)
    with pytest.raises(RuntimeError, match="no model shards"):
        ckpt.download_dreamdojo_checkpoint("2B_pretrain")


def test_converts_keeps_ema_and_prunes_shards(monkeypatch, tmp_path):
    import huggingface_hub

    home = tmp_path / "wk"
    monkeypatch.setenv("WORLDKERNELS_HOME", str(home))
    _patch_latest(monkeypatch, tmp_path, "iter_000140000")

    def fake_snap(*a, local_dir=None, **k):
        md = Path(local_dir, "2B_pretrain", "iter_000140000", "model")
        md.mkdir(parents=True)
        (md / "__0_0.distcp").write_bytes(b"x")
        (md / ".metadata").write_bytes(b"x")
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snap)
    monkeypatch.setattr(
        ckpt,
        "_consolidate_dcp",
        lambda model_dir, scratch: {
            "net_ema.w": torch.ones(4, dtype=torch.float32),
            "net.w": torch.zeros(4, dtype=torch.float32),
        },
    )

    out = ckpt.download_dreamdojo_checkpoint("2B_pretrain")

    out_dir = home / "dreamdojo" / "2B_pretrain" / "iter_000140000"
    assert out == str(out_dir / "model_ema_bf16.pt")
    assert Path(out).exists()
    assert not (out_dir / "_dcp").exists(), "raw DCP shards must be pruned after conversion"

    saved = torch.load(out, map_location="cpu", weights_only=False)
    assert set(saved) == {"net.w"}
    assert saved["net.w"].dtype == torch.bfloat16


def test_existing_pt_short_circuits(monkeypatch, tmp_path):
    import huggingface_hub

    home = tmp_path / "wk"
    monkeypatch.setenv("WORLDKERNELS_HOME", str(home))
    _patch_latest(monkeypatch, tmp_path, "iter_000140000")
    pt = home / "dreamdojo" / "2B_pretrain" / "iter_000140000" / "model_ema_bf16.pt"
    pt.parent.mkdir(parents=True)
    pt.write_bytes(b"cached")
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("must not re-download")
    )
    assert ckpt.download_dreamdojo_checkpoint("2B_pretrain") == str(pt)
