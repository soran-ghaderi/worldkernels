r"""Tests for worldkernels/worlds/adapters/dreamdojo/checkpoint.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from worldkernels.worlds.adapters.dreamdojo import checkpoint as ckpt_mod
from worldkernels.worlds.adapters.dreamdojo.checkpoint import (
    HF_REPO,
    download_dreamdojo_checkpoint,
)


def _mk_layout(tmp_path: Path, ckpt_dir_name: str, iter_name: str = "iter_000001") -> Path:
    base = tmp_path / "snap" / ckpt_dir_name
    base.mkdir(parents=True)
    (base / "latest_checkpoint.txt").write_text(iter_name)
    iter_dir = base / iter_name
    iter_dir.mkdir()
    (iter_dir / "model").mkdir()
    return base


class TestDownload:
    def test_repo_constant(self):
        assert HF_REPO == "nvidia/DreamDojo"

    def test_returns_existing_pt(self, tmp_path, monkeypatch):
        base = _mk_layout(tmp_path, "2B_pretrain")
        iter_dir = base / "iter_000001"
        existing = iter_dir / "model_ema_bf16.pt"
        existing.write_bytes(b"")

        snapshot = MagicMock(return_value=str(tmp_path / "snap"))
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", snapshot, raising=False
        )
        path = download_dreamdojo_checkpoint("2B_pretrain")
        assert Path(path) == existing
        snapshot.assert_called_once_with(
            "nvidia/DreamDojo",
            allow_patterns=[
                "2B_pretrain/**/model/*",
                "2B_pretrain/latest_checkpoint.txt",
            ],
            repo_type="model",
        )

    def test_converts_dcp_to_pt(self, tmp_path, monkeypatch):
        base = _mk_layout(tmp_path, "2B_pretrain")
        iter_dir = base / "iter_000001"

        snapshot = MagicMock(return_value=str(tmp_path / "snap"))
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", snapshot, raising=False
        )

        def fake_dcp(model_dir, full_pt_path):
            torch.save(
                {
                    "net_ema.weight": torch.ones(2, 2, dtype=torch.float32),
                    "net_ema.bias": torch.ones(2, dtype=torch.bfloat16),
                    "net.other_unrelated": torch.ones(2, 2, dtype=torch.float32),
                },
                full_pt_path,
            )

        monkeypatch.setattr(
            "torch.distributed.checkpoint.format_utils.dcp_to_torch_save", fake_dcp
        )

        path = download_dreamdojo_checkpoint("2B_pretrain")
        out = Path(path)
        assert out.exists()
        assert (iter_dir / "model.pt").exists() is False  # cleaned up
        state = torch.load(out, map_location="cpu", weights_only=False)
        assert "net.weight" in state
        assert "net.bias" in state
        assert "net_ema.weight" not in state
        assert "net.other_unrelated" not in state
        assert state["net.weight"].dtype == torch.bfloat16
        assert state["net.bias"].dtype == torch.bfloat16

    def test_default_ckpt_dir(self, tmp_path, monkeypatch):
        base = _mk_layout(tmp_path, "2B_pretrain")
        (base / "iter_000001" / "model_ema_bf16.pt").write_bytes(b"")
        snapshot = MagicMock(return_value=str(tmp_path / "snap"))
        monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot, raising=False)
        path = download_dreamdojo_checkpoint()
        assert "2B_pretrain" in path
