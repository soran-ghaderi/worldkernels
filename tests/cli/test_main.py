r"""Tests for worldkernels/cli/main.py (argparse dispatch shell)."""

from __future__ import annotations

import os

import pytest

from worldkernels.cli.main import app, build_parser


@pytest.fixture(autouse=True)
def _reset_ui_state():
    import logging

    wk_log = logging.getLogger("worldkernels")
    saved = (wk_log.handlers[:], wk_log.propagate, wk_log.level)
    yield
    from worldkernels.ui.verbosity import OutputMode, Verbosity, set_output_mode, set_verbosity

    set_verbosity(Verbosity.NORMAL)
    set_output_mode(OutputMode.AUTO)
    os.environ.pop("WORLDKERNELS_QUIET", None)
    wk_log.handlers[:] = saved[0]
    wk_log.propagate = saved[1]
    wk_log.setLevel(saved[2])


@pytest.fixture
def serve_recorder(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "worldkernels.cli.commands.serve.run_serve",
        lambda **kwargs: calls.append(kwargs),
    )
    return calls


class TestVersion:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as ei:
            app(["--version"])
        assert ei.value.code == 0
        assert "worldkernels" in capsys.readouterr().out

    def test_short_form(self, capsys):
        with pytest.raises(SystemExit) as ei:
            app(["-V"])
        assert ei.value.code == 0


class TestDispatch:
    def test_serve_defaults(self, serve_recorder):
        app(["serve"])
        assert len(serve_recorder) == 1
        call = serve_recorder[0]
        assert call["host"] == "0.0.0.0"
        assert call["port"] == 8000
        assert call["model"] is None

    def test_serve_flags(self, serve_recorder):
        app(["serve", "dummy", "--port", "9000", "-k", "sk-1"])
        call = serve_recorder[0]
        assert call["model"] == "dummy"
        assert call["port"] == 9000
        assert call["api_key"] == "sk-1"

    def test_serve_underscore_alias(self, serve_recorder):
        app(["serve", "--max_sessions", "2"])
        assert serve_recorder[0]["max_sessions"] == 2

    def test_serve_frontend_flags(self, serve_recorder):
        app(
            [
                "serve",
                "--root-path",
                "/api",
                "--uvicorn-log-level",
                "warning",
                "--allowed-origins",
                "http://a,http://b",
                "--disable-access-log",
                "--allow-credentials",
                "--ssl-keyfile",
                "key.pem",
            ]
        )
        call = serve_recorder[0]
        assert call["root_path"] == "/api"
        assert call["uvicorn_log_level"] == "warning"
        assert call["allowed_origins"] == ["http://a", "http://b"]
        assert call["disable_access_log"] is True
        assert call["allow_credentials"] is True
        assert call["ssl_keyfile"] == "key.pem"

    def test_serve_frontend_defaults(self, serve_recorder):
        app(["serve"])
        call = serve_recorder[0]
        assert call["allowed_origins"] == ["*"]
        assert call["uvicorn_log_level"] == "info"
        assert call["root_path"] == ""

    def test_serve_set_overrides(self, serve_recorder):
        app(["serve", "--set", "teacache=on,attention_backend=sdpa"])
        assert serve_recorder[0]["overrides"] == {
            "teacache": True,
            "attention_backend": "sdpa",
        }

    def test_run_defaults(self, monkeypatch):
        calls: list[dict] = []
        monkeypatch.setattr(
            "worldkernels.cli.commands.run.run_session",
            lambda **kwargs: calls.append(kwargs),
        )
        app(["run"])
        assert calls[0]["model"] == "dummy"
        assert calls[0]["steps"] == 10
        assert calls[0]["decode"] is True

    def test_rm_requires_model(self):
        with pytest.raises(SystemExit) as ei:
            app(["rm"])
        assert ei.value.code == 2

    def test_unknown_command_exits_2(self):
        with pytest.raises(SystemExit) as ei:
            app(["frobnicate"])
        assert ei.value.code == 2


class TestBenchFlatForm:
    def test_bench_latency(self, monkeypatch):
        calls: list[dict] = []
        monkeypatch.setattr(
            "worldkernels.cli.commands.bench.run_latency",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        app(["bench", "latency", "--world", "dummy", "--steps", "5"])
        assert len(calls) == 1

    def test_bench_requires_subcommand(self):
        with pytest.raises(SystemExit) as ei:
            app(["bench"])
        assert ei.value.code == 2


class TestGlobalFlags:
    def test_quiet_before_verb(self, serve_recorder):
        from worldkernels.ui.verbosity import Verbosity, get_verbosity

        app(["-q", "serve"])
        assert get_verbosity() == Verbosity.QUIET

    def test_quiet_after_verb(self, serve_recorder):
        from worldkernels.ui.verbosity import Verbosity, get_verbosity

        app(["serve", "-q"])
        assert get_verbosity() == Verbosity.QUIET

    def test_stacked_verbose_is_debug(self, serve_recorder):
        from worldkernels.ui.verbosity import Verbosity, get_verbosity

        app(["serve", "-vv"])
        assert get_verbosity() == Verbosity.DEBUG

    def test_output_json(self, serve_recorder):
        from worldkernels.ui.verbosity import OutputMode, get_output_mode

        app(["--output", "json", "serve"])
        assert get_output_mode() == OutputMode.JSON

    def test_bad_output_exits_2(self):
        with pytest.raises(SystemExit) as ei:
            app(["--output", "bogus", "serve"])
        assert ei.value.code == 2


class TestEngineFlags:
    def test_dedicated_flags_collected(self, serve_recorder):
        app(["serve", "--teacache", "--tensor-parallel-size", "2"])
        assert serve_recorder[0]["overrides"] == {
            "teacache": True,
            "parallel.tensor_parallel_size": 2,
        }

    def test_dedicated_flag_beats_set(self, serve_recorder):
        app(["serve", "--set", "teacache=off", "--teacache"])
        assert serve_recorder[0]["overrides"]["teacache"] is True

    def test_no_flags_means_no_overrides(self, serve_recorder):
        app(["serve"])
        assert serve_recorder[0]["overrides"] is None
        assert serve_recorder[0]["config_file"] is None

    def test_set_numeric_coercion(self, serve_recorder):
        app(["serve", "--set", "scheduler.max_batch_size=4"])
        assert serve_recorder[0]["overrides"] == {"scheduler.max_batch_size": 4}

    def test_bench_latency_engine_flags(self, monkeypatch):
        calls: list[dict] = []
        monkeypatch.setattr(
            "worldkernels.cli.commands.bench.run_latency",
            lambda *args, **kwargs: calls.append(kwargs),
        )
        app(["bench", "latency", "--world", "dummy", "--no-torch-compile"])
        assert calls[0]["overrides"] == {"torch_compile": False}


class TestConfigFile:
    def test_engine_keys_pass_through(self, serve_recorder, tmp_path):
        path = tmp_path / "wk.yaml"
        path.write_text("teacache: true\n")
        app(["serve", "--config", str(path)])
        assert serve_recorder[0]["config_file"] == str(path)
        assert serve_recorder[0]["overrides"] is None

    def test_frontend_keys_become_flags(self, serve_recorder, tmp_path):
        path = tmp_path / "wk.yaml"
        path.write_text("port: 9000\nhost: 127.0.0.1\n")
        app(["serve", "--config", str(path)])
        assert serve_recorder[0]["port"] == 9000
        assert serve_recorder[0]["host"] == "127.0.0.1"

    def test_explicit_flag_beats_file(self, serve_recorder, tmp_path):
        path = tmp_path / "wk.yaml"
        path.write_text("port: 9000\n")
        app(["serve", "--config", str(path), "--port", "7000"])
        assert serve_recorder[0]["port"] == 7000

    def test_explicit_flag_before_config_still_wins(self, serve_recorder, tmp_path):
        path = tmp_path / "wk.yaml"
        path.write_text("port: 9000\n")
        app(["serve", "--port", "7000", "--config", str(path)])
        assert serve_recorder[0]["port"] == 7000

    def test_frontend_bool_from_file(self, serve_recorder, tmp_path):
        path = tmp_path / "wk.yaml"
        path.write_text("no_fetch: true\n")
        app(["serve", "--config", str(path)])
        assert serve_recorder[0]["allow_fetch"] is False

    def test_unknown_frontend_key_exits_2(self, tmp_path):
        path = tmp_path / "wk.yaml"
        path.write_text("bogus_flag: 1\n")
        with pytest.raises(SystemExit) as ei:
            app(["serve", "--config", str(path)])
        assert ei.value.code == 2

    def test_config_show_skips_frontend_keys(self, monkeypatch, tmp_path, capsys):
        path = tmp_path / "wk.yaml"
        path.write_text("port: 9000\nteacache: true\n")
        app(["config-show", "--config", str(path), "--json"])
        import json

        data = json.loads(capsys.readouterr().out)
        assert data["teacache"]["value"] is True
        assert data["teacache"]["source"] == f"config:{path}"


class TestHelpSearch:
    def test_subcommand_keyword_search(self, capsys):
        with pytest.raises(SystemExit) as ei:
            app(["serve", "--help=fetch"])
        assert ei.value.code == 0
        out = capsys.readouterr().out
        assert "--no-fetch" in out
        assert "--port" not in out

    def test_engine_group_help(self, capsys):
        with pytest.raises(SystemExit) as ei:
            app(["serve", "--help=parallelconfig"])
        assert ei.value.code == 0
        out = capsys.readouterr().out
        assert "--tensor-parallel-size" in out
        assert "--teacache" not in out

    def test_engine_keyword_search(self, capsys):
        with pytest.raises(SystemExit):
            app(["serve", "--help=teacache"])
        out = capsys.readouterr().out
        assert "--teacache" in out
        assert "--tensor-parallel-size" not in out

    def test_frontend_group_help(self, capsys):
        with pytest.raises(SystemExit):
            app(["serve", "--help=frontend"])
        out = capsys.readouterr().out
        assert "--ssl-keyfile" in out
        assert "--root-path" in out
        assert "--teacache" not in out

    def test_root_help_lists_verbs(self, capsys):
        with pytest.raises(SystemExit):
            app(["--help"])
        out = capsys.readouterr().out
        for verb in ("serve", "run", "pull", "models", "bench", "collect-env"):
            assert verb in out


class TestBuildParser:
    def test_all_verbs_registered(self):
        parser = build_parser()
        subactions = [a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"]
        assert len(subactions) == 1
        verbs = set(subactions[0].choices)
        assert verbs >= {
            "serve",
            "run",
            "pull",
            "models",
            "rm",
            "collect-env",
            "config-show",
            "inspect",
            "bench",
            "plugins",
        }
