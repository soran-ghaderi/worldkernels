r"""Tests for tests/fixtures/cosmos_reference.py (capture script).

Capture itself requires GPU + cosmos_predict2 install — that path is exercised
under `pytest --regen-fixtures` on a GPU host. These tests cover pure helpers
and CLI argument parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.fixtures import cosmos_reference as cr


class TestCaptureConfig:
    def test_defaults(self):
        c = cr.CaptureConfig()
        assert c.adapter == "dreamdojo"
        assert c.variant == "2b_pretrain"
        assert c.height == 240
        assert c.width == 320

    def test_name_string(self):
        c = cr.CaptureConfig(seed=42, num_steps=5)
        assert "dreamdojo_2b_pretrain" in c.name
        assert "seed42" in c.name
        assert "steps5" in c.name

    def test_output_dir_under_root(self):
        c = cr.CaptureConfig(output_root=Path("/tmp/x"))
        assert str(c.output_dir).startswith("/tmp/x/")


class TestParseDtype:
    @pytest.mark.parametrize(
        "s,dtype",
        [
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
            ("float32", torch.float32),
        ],
    )
    def test_known(self, s, dtype):
        assert cr._parse_dtype(s) == dtype

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            cr._parse_dtype("int8")


class TestColorBars:
    def test_shape_and_range(self):
        f = cr._make_init_frame(16, 24)
        assert f.shape == (3, 16, 24)
        assert f.dtype == torch.float32
        assert ((f == 0.0) | (f == 1.0)).all()

    def test_deterministic(self):
        assert torch.equal(cr._make_init_frame(8, 8), cr._make_init_frame(8, 8))


class TestDeterministicJoints:
    def test_shape(self):
        joints = cr._deterministic_joints(chunk_size=4, action_dim=7, seed=1)
        assert isinstance(joints, list)
        assert len(joints) == 4
        assert all(len(row) == 7 for row in joints)

    def test_reproducible(self):
        a = cr._deterministic_joints(4, 7, 5)
        b = cr._deterministic_joints(4, 7, 5)
        assert a == b

    def test_different_seeds_diverge(self):
        a = cr._deterministic_joints(4, 7, 1)
        b = cr._deterministic_joints(4, 7, 2)
        assert a != b


class TestCaptureBuffer:
    def test_record_and_collision_detection(self):
        buf = cr._CaptureBuffer()
        t = torch.randn(2)
        buf.record("a", t)
        assert "a" in buf.tensors
        assert torch.equal(buf.tensors["a"], t)
        with pytest.raises(KeyError, match="collision"):
            buf.record("a", t)

    def test_meta_populated(self):
        buf = cr._CaptureBuffer()
        buf.record("x", torch.zeros(2, 3, dtype=torch.float32), note="hi")
        assert buf.meta["x"]["shape"] == [2, 3]
        assert buf.meta["x"]["dtype"] == "torch.float32"
        assert buf.meta["x"]["note"] == "hi"


class TestArgParser:
    def test_defaults(self):
        cfg = cr._parse_args([])
        assert cfg.adapter == "dreamdojo"
        assert cfg.variant == "2b_pretrain"

    def test_overrides(self):
        cfg = cr._parse_args(
            [
                "--adapter",
                "cosmos",
                "--height",
                "128",
                "--width",
                "256",
                "--pixel-frames",
                "9",
                "--dtype",
                "float16",
                "--seed",
                "777",
                "--num-steps",
                "12",
                "--guidance",
                "2.5",
                "--action-dim",
                "32",
                "--chunk-size",
                "16",
                "--output-root",
                "/tmp/foo",
            ]
        )
        assert cfg.adapter == "cosmos"
        assert cfg.height == 128
        assert cfg.width == 256
        assert cfg.pixel_frames == 9
        assert cfg.dtype_str == "float16"
        assert cfg.seed == 777
        assert cfg.num_steps == 12
        assert cfg.guidance == 2.5
        assert cfg.action_dim == 32
        assert cfg.chunk_size == 16
        assert cfg.output_root == Path("/tmp/foo")


class TestWriteSafetensors:
    def test_writes_each_tensor(self, tmp_path):
        tensors = {"a": torch.zeros(2, 2), "b": torch.ones(3)}
        cr._write_safetensors(tmp_path, tensors)
        assert (tmp_path / "a.safetensors").exists()
        assert (tmp_path / "b.safetensors").exists()


class TestWriteManifest:
    def test_writes_json(self, tmp_path):
        cfg = cr.CaptureConfig(output_root=tmp_path)
        cr._write_manifest(tmp_path, cfg, {"a": {"shape": [1]}}, (0, 1), 3)
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        import json

        data = json.loads(manifest_path.read_text())
        assert data["fixture_version"] == cr.FIXTURE_VERSION
        assert data["dit_blocks_total"] == 3
        assert data["dit_blocks_hooked"] == [0, 1]
        assert "tensors" in data


class TestCaptureUnknownAdapter:
    def test_raises_value_error(self, monkeypatch):
        cfg = cr.CaptureConfig(adapter="bogus", output_root=Path("/tmp"))

        monkeypatch.setattr(
            "worldkernels.worlds.pipelines.cosmos_predict2.deps.ensure_cosmos_predict2",
            lambda: None,
        )
        with pytest.raises(ValueError, match="unknown adapter"):
            cr.capture(cfg)


class TestMainHelp:
    def test_main_with_unknown_arg_exits(self, capsys):
        with pytest.raises(SystemExit):
            cr.main(["--bad"])
