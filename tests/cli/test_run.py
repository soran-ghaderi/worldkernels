r"""Tests for worldkernels/cli/run.py."""

from __future__ import annotations

import pytest

from worldkernels.cli.run import _raw_to_arrays, _save_frames, _save_video, run_session


class TestRunSession:
    def test_invalid_output_format_raises(self):
        with pytest.raises(ValueError, match="output-format"):
            run_session(model="dummy", output_format="bad", device="cpu")

    def test_default_run_no_output_dir(self, capsys):
        run_session(model="dummy", steps=2, height=32, width=32, device="cpu")
        out = capsys.readouterr().out
        assert "Running 2 steps" in out
        assert "Done. 2 steps" in out

    def test_output_dir_saves_frames(self, tmp_path, capsys):
        run_session(
            model="dummy",
            steps=2,
            height=32,
            width=32,
            device="cpu",
            output_dir=str(tmp_path),
            output_format="frames",
        )
        pngs = list(tmp_path.glob("frame_*.png"))
        assert len(pngs) >= 1

    def test_output_video_format(self, tmp_path):
        pytest.importorskip("imageio_ffmpeg")
        run_session(
            model="dummy",
            steps=2,
            height=32,
            width=32,
            device="cpu",
            output_dir=str(tmp_path),
            output_format="video",
        )
        assert (tmp_path / "output.mp4").exists()

    def test_output_both(self, tmp_path):
        pytest.importorskip("imageio_ffmpeg")
        run_session(
            model="dummy",
            steps=2,
            height=32,
            width=32,
            device="cpu",
            output_dir=str(tmp_path),
            output_format="both",
        )
        assert (tmp_path / "output.mp4").exists()
        assert any(tmp_path.glob("frame_*.png"))

    def test_decode_false_skips_frames(self, tmp_path):
        run_session(
            model="dummy",
            steps=2,
            height=32,
            width=32,
            device="cpu",
            output_dir=str(tmp_path),
            output_format="frames",
            decode=False,
        )
        assert list(tmp_path.glob("frame_*.png")) == []


class TestHelpers:
    def test_save_frames_writes_png(self, tmp_path):
        raw = b"\x00\x00\x00" * 64 * 64
        n = _save_frames([raw], tmp_path, 64, 64, 5)
        assert n == 1
        assert (tmp_path / "frame_00005.png").exists()

    def test_save_frames_ignores_non_bytes(self, tmp_path):
        n = _save_frames([42, "str", 3.14], tmp_path, 8, 8, 0)
        assert n == 0

    def test_save_frames_non_list_input(self, tmp_path):
        n = _save_frames(b"single", tmp_path, 8, 8, 0)
        assert n == 0

    def test_raw_to_arrays_filters_non_bytes(self):
        raw = b"\x00\x01\x02" * 8 * 8
        arrs = _raw_to_arrays([raw, 42], 8, 8)
        assert len(arrs) == 1
        assert arrs[0].shape == (8, 8, 3)

    def test_raw_to_arrays_non_list(self):
        assert _raw_to_arrays(b"x", 8, 8) == []

    def test_save_video_writes_file(self, tmp_path):
        pytest.importorskip("imageio_ffmpeg")
        import numpy as np

        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
        path = tmp_path / "out.mp4"
        _save_video(frames, path, fps=12, codec="libx264")
        assert path.exists()
