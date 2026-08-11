r"""Tests for worldkernels/cli/parser.py (FlexibleArgumentParser)."""

from __future__ import annotations

import argparse
import json

import pytest

from worldkernels.cli.parser import (
    WK_SUBCMD_EPILOG,
    FlexibleArgumentParser,
    SortedHelpFormatter,
    human_readable_int,
)


def make_parser() -> FlexibleArgumentParser:
    p = FlexibleArgumentParser(prog="wk-test", formatter_class=SortedHelpFormatter)
    cache = p.add_argument_group("CacheConfig")
    cache.add_argument("--block-frames", type=int, default=16, help="Frames per latent block.")
    cache.add_argument("--gpu-memory-fraction", type=float, default=0.8)
    runtime = p.add_argument_group("RuntimeConfig")
    runtime.add_argument("--teacache", action=argparse.BooleanOptionalAction, default=False)
    runtime.add_argument("--attention-backend", choices=("auto", "flash", "sdpa"), default="auto")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--compilation-config", type=json.loads, default=None)
    return p


class TestHumanReadableInt:
    def test_plain(self):
        assert human_readable_int("256") == 256

    def test_decimal_suffixes(self):
        assert human_readable_int("1k") == 1_000
        assert human_readable_int("25.6k") == 25_600
        assert human_readable_int("2m") == 2_000_000
        assert human_readable_int("1g") == 1_000_000_000

    def test_binary_suffixes(self):
        assert human_readable_int("16K") == 16_384
        assert human_readable_int("2M") == 2 * 2**20
        assert human_readable_int("1G") == 2**30

    def test_decimal_on_binary_suffix_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            human_readable_int("2.5K")

    def test_garbage_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            human_readable_int("12q")
        with pytest.raises(argparse.ArgumentTypeError):
            human_readable_int("k")


class TestFlagForms:
    def test_underscore_alias(self):
        args = make_parser().parse_args(["--block_frames", "32"])
        assert args.block_frames == 32

    def test_equals_form(self):
        args = make_parser().parse_args(["--block-frames=32"])
        assert args.block_frames == 32

    def test_underscore_equals_form(self):
        args = make_parser().parse_args(["--gpu_memory_fraction=0.5"])
        assert args.gpu_memory_fraction == 0.5

    def test_bool_optional_action(self):
        assert make_parser().parse_args(["--teacache"]).teacache is True
        assert make_parser().parse_args(["--no-teacache"]).teacache is False


class TestDottedJsonArgs:
    def test_dotted_keys_build_dict(self):
        args = make_parser().parse_args(
            ["--compilation-config.mode", "2", "--compilation-config.passes.fuse", "true"]
        )
        assert args.compilation_config == {"mode": 2, "passes": {"fuse": True}}

    def test_dotted_equals_form(self):
        args = make_parser().parse_args(["--compilation-config.mode=2"])
        assert args.compilation_config == {"mode": 2}

    def test_plus_appends_list(self):
        args = make_parser().parse_args(
            ["--compilation-config.tags+", "a", "--compilation-config.tags+", "b"]
        )
        assert args.compilation_config == {"tags": ["a", "b"]}

    def test_value_coercion_chain(self):
        args = make_parser().parse_args(
            ["--compilation-config.n", "25.6k", "--compilation-config.name", "plain"]
        )
        assert args.compilation_config == {"n": 25_600, "name": "plain"}

    def test_whole_json_still_works(self):
        args = make_parser().parse_args(["--compilation-config", '{"mode": 1}'])
        assert args.compilation_config == {"mode": 1}


class TestHelpSystem:
    def _help_output(self, argv: list[str], capsys) -> str:
        with pytest.raises(SystemExit) as ei:
            make_parser().parse_args(argv)
        assert ei.value.code == 0
        return capsys.readouterr().out

    def test_help_all_shows_everything(self, capsys):
        out = self._help_output(["--help=all"], capsys)
        assert "--block-frames" in out
        assert "--teacache" in out
        assert "--port" in out

    def test_help_group_case_insensitive(self, capsys):
        out = self._help_output(["--help=cacheconfig"], capsys)
        assert "--block-frames" in out
        assert "--teacache" not in out

    def test_help_group_underscore_normalized(self, capsys):
        out = self._help_output(["--help=cache_config"], capsys)
        assert "--block-frames" in out

    def test_help_keyword_search(self, capsys):
        out = self._help_output(["--help=teacache"], capsys)
        assert "--teacache" in out
        assert "--port" not in out

    def test_help_no_match_suggests_all(self, capsys):
        out = self._help_output(["--help=zzz_nothing"], capsys)
        assert "--help=all" in out

    def test_defaults_shown_in_help(self, capsys):
        out = self._help_output(["--help=all"], capsys)
        assert "8000" in out

    def test_epilog_advertises_help_system(self):
        assert "--help=" in WK_SUBCMD_EPILOG


@pytest.fixture
def parser_log_records():
    import logging

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    plog = logging.getLogger("worldkernels.cli.parser")
    old_level = plog.level
    plog.setLevel(logging.WARNING)
    plog.addHandler(handler)
    yield records
    plog.removeHandler(handler)
    plog.setLevel(old_level)


class TestDeprecation:
    def test_deprecated_flag_warns_once(self, parser_log_records):
        FlexibleArgumentParser._warned_deprecated.discard("--old-flag")
        p = FlexibleArgumentParser(prog="wk-test")
        p.add_argument("--old-flag", type=int, deprecated=True)
        p.parse_args(["--old-flag", "1"])
        p.parse_args(["--old-flag", "2"])
        warnings = [r for r in parser_log_records if "old-flag" in r.getMessage()]
        assert len(warnings) == 1

    def test_non_deprecated_flag_silent(self, parser_log_records):
        p = FlexibleArgumentParser(prog="wk-test")
        p.add_argument("--new-flag", type=int)
        p.parse_args(["--new-flag", "1"])
        assert not [r for r in parser_log_records if "new-flag" in r.getMessage()]


class TestSortedHelpFormatter:
    def test_options_sorted(self, capsys):
        p = FlexibleArgumentParser(prog="wk-test", formatter_class=SortedHelpFormatter)
        p.add_argument("--zebra")
        p.add_argument("--apple")
        with pytest.raises(SystemExit):
            p.parse_args(["--help=all"])
        out = capsys.readouterr().out
        body = out[out.index("options:") :]
        assert body.index("--apple") < body.index("--zebra")
