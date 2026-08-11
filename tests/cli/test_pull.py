r"""Tests for worldkernels/cli/commands/pull.py (models / rm handlers)."""

from __future__ import annotations

from worldkernels.cli.commands.pull import _human, run_models, run_rm


class TestRunModels:
    def test_all_lists_hub_entries(self, capsys):
        run_models(show_all=True)
        out = capsys.readouterr().out
        assert "dummy" in out
        assert "dreamdojo" in out

    def test_empty_cache_hints_pull(self, monkeypatch, capsys):
        monkeypatch.setattr("worldkernels.bootstrap.cache.list_manifests", lambda: [])
        run_models(show_all=False)
        out = capsys.readouterr().out
        assert "no models cached locally" in out
        assert "pull" in out


class TestRunRm:
    def test_nothing_to_remove(self, monkeypatch, capsys):
        monkeypatch.setattr("worldkernels.bootstrap.cache.remove_manifest", lambda m, v: False)
        monkeypatch.setattr("worldkernels.runtime.envs.remove_env", lambda m: False)
        run_rm("no_such_local_model")
        out = capsys.readouterr().out
        assert "nothing to remove" in out

    def test_removes_manifest(self, monkeypatch, capsys):
        monkeypatch.setattr("worldkernels.bootstrap.cache.remove_manifest", lambda m, v: True)
        monkeypatch.setattr("worldkernels.runtime.envs.remove_env", lambda m: False)
        run_rm("no_such_local_model")
        out = capsys.readouterr().out
        assert "removed manifest: no_such_local_model" in out


class TestHuman:
    def test_units(self):
        assert _human(512) == "512.0B"
        assert _human(2048) == "2.0KB"
        assert _human(3 * 1024**3) == "3.0GB"
