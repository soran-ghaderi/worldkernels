r"""Tests for worldkernels/cli/config_args.py (dataclass flag groups)."""

from __future__ import annotations

import pytest

from worldkernels.cli.config_args import ENGINE_GROUPS, add_engine_args, collect_overrides
from worldkernels.cli.parser import FlexibleArgumentParser


def make_parser() -> FlexibleArgumentParser:
    p = FlexibleArgumentParser(prog="wk-test")
    add_engine_args(p)
    return p


def all_option_strings(p: FlexibleArgumentParser) -> set[str]:
    return {opt for a in p._actions for opt in a.option_strings}


class TestFlagGeneration:
    def test_group_titles(self):
        titles = {g.title for g in make_parser()._action_groups}
        assert {"RuntimeConfig", "ParallelConfig", "CacheConfig", "SchedulerConfig"} <= titles

    def test_bool_flags_have_negative_form(self):
        opts = all_option_strings(make_parser())
        assert "--teacache" in opts
        assert "--no-teacache" in opts

    def test_nested_flags_present(self):
        opts = all_option_strings(make_parser())
        assert "--tensor-parallel-size" in opts
        assert "--block-frames" in opts
        assert "--max-batch-size" in opts

    def test_cli_owned_fields_excluded(self):
        opts = all_option_strings(make_parser())
        assert "--device" not in opts
        assert "--max-sessions" not in opts

    def test_enum_choices_enforced(self):
        with pytest.raises(SystemExit) as ei:
            make_parser().parse_args(["--dtype", "garbage"])
        assert ei.value.code == 2

    def test_help_shows_defaults(self, capsys):
        with pytest.raises(SystemExit):
            make_parser().parse_args(["--help=runtimeconfig"])
        assert "(default: auto)" in capsys.readouterr().out


class TestCollectOverrides:
    def test_only_explicit_flags_collected(self):
        args = make_parser().parse_args(["--teacache", "--tensor-parallel-size", "2"])
        assert collect_overrides(args) == {"teacache": True, "parallel.tensor_parallel_size": 2}

    def test_empty_when_nothing_passed(self):
        assert collect_overrides(make_parser().parse_args([])) == {}

    def test_negative_bool(self):
        args = make_parser().parse_args(["--no-torch-compile"])
        assert collect_overrides(args) == {"torch_compile": False}

    def test_nested_dotted_dests(self):
        args = make_parser().parse_args(
            ["--block-frames", "8", "--max-batch-size", "4", "--policy", "priority"]
        )
        assert collect_overrides(args) == {
            "cache.block_frames": 8,
            "scheduler.max_batch_size": 4,
            "scheduler.policy": "priority",
        }

    def test_underscore_alias(self):
        args = make_parser().parse_args(["--tensor_parallel_size", "4"])
        assert collect_overrides(args) == {"parallel.tensor_parallel_size": 4}

    def test_typed_values(self):
        args = make_parser().parse_args(["--gpu-memory-fraction", "0.5"])
        assert collect_overrides(args) == {"cache.gpu_memory_fraction": 0.5}

    def test_roundtrips_into_resolver(self):
        from worldkernels.config.profiles import resolve_runtime_config

        args = make_parser().parse_args(["--teacache", "--ring-degree", "2"])
        cfg, sources = resolve_runtime_config(cli_overrides=collect_overrides(args), env={})
        assert cfg.teacache is True
        assert cfg.parallel.ring_degree == 2
        assert sources["parallel.ring_degree"] == "cli:--ring-degree"


class TestEngineGroups:
    def test_prefixes_match_runtime_config_fields(self):
        from worldkernels.config.runtime import RuntimeConfig

        nested = {prefix for _, _, prefix in ENGINE_GROUPS if prefix}
        annotations = set(RuntimeConfig.__dataclass_fields__)
        assert nested <= annotations
