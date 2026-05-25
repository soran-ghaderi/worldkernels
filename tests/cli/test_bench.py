r"""Tests for worldkernels/cli/bench.py."""

from __future__ import annotations

from worldkernels.cli.bench import (
    bench_env,
    run_latency,
    run_startup,
    run_throughput,
    run_vram,
)


class TestBenchEnv:
    def test_yields_engine_and_sessions(self):
        with bench_env("dummy", "cpu") as (wk, sessions):
            assert len(sessions) == 1
            assert sessions[0].world_id == "dummy"
        assert "dummy" not in wk.list_worlds()


class TestRunLatency:
    def test_prints_summary(self, capsys):
        run_latency("dummy", 5, 32, 32, "cpu")
        out = capsys.readouterr().out
        for tag in ("mean", "p50", "p99", "min", "max"):
            assert tag in out


class TestRunThroughput:
    def test_prints_summary(self, capsys):
        run_throughput("dummy", 2, 3, 32, 32, "cpu")
        out = capsys.readouterr().out
        assert "total steps: 6" in out
        assert "throughput" in out


class TestRunVram:
    def test_prints_estimates(self, capsys):
        run_vram("dummy", "cpu", "32x32,64x64")
        out = capsys.readouterr().out
        assert "32x32" in out
        assert "64x64" in out
        assert "VRAM (MB)" in out


class TestRunStartup:
    def test_prints_timing(self, capsys):
        run_startup("dummy", "cpu")
        out = capsys.readouterr().out
        assert "engine init" in out
        assert "load+warmup" in out


class TestRunProfile:
    def test_writes_trace(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        from worldkernels.cli.bench import run_profile

        run_profile("dummy", steps=2, height=16, width=16, device="cpu", output="wk_prof")
        assert (tmp_path / "wk_prof.json").exists()
        out = capsys.readouterr().out
        assert "Trace written" in out
