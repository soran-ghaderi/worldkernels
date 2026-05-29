r"""WorldKernels CLI: vLLM-style positional verbs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Annotated, Union

import tyro

_PositionalModel = Annotated[
    str,
    tyro.conf.Positional,
    tyro.conf.arg(help="Model: short alias, HF repo id, HF URL, or local checkpoint path."),
]


@dataclass
class Serve:
    r"""Start the WorldKernels HTTP/WebSocket server.

    With a model arg, the model is bootstrapped (deps, weights) and pre-loaded;
    without one, the server starts empty and models load on first request.
    """

    model: Annotated[
        str | None,
        tyro.conf.Positional,
        tyro.conf.arg(help="Model to pre-load (optional)."),
    ] = None
    host: str = "0.0.0.0"
    port: int = 8000
    max_sessions: int = 4
    api_key: Annotated[str | None, tyro.conf.arg(aliases=("-k",))] = None
    device: str = "cuda"
    variant: str | None = None
    ckpt_path: str | None = None
    profile: str | None = None
    set_overrides: Annotated[
        str | None,
        tyro.conf.arg(
            name="set",
            help="Comma-separated runtime overrides (e.g. teacache=off,attention_backend=sdpa).",
        ),
    ] = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    no_fetch: bool = False
    quiet: Annotated[bool, tyro.conf.arg(aliases=("-q",))] = False

    def run(self) -> None:
        from worldkernels.cli.serve import run_serve

        run_serve(
            host=self.host,
            port=self.port,
            max_sessions=self.max_sessions,
            api_key=self.api_key,
            device=self.device,
            model=self.model,
            variant=self.variant,
            ckpt_path=self.ckpt_path,
            model_kwargs=_extra_kwargs(self.num_inference_steps, self.guidance_scale),
            allow_fetch=not self.no_fetch,
            quiet=self.quiet,
            profile=self.profile,
            overrides=_parse_set(self.set_overrides),
        )


@dataclass
class Run:
    r"""Headless run: bootstrap a model, step N times, optionally save frames/video."""

    model: _PositionalModel = "dummy"
    steps: int = 10
    action_type: str = "null"
    height: int = 480
    width: int = 848
    device: str = "cuda"
    seed: int = 0
    output_dir: Annotated[str | None, tyro.conf.arg(aliases=("-o",))] = None
    output_format: str = "frames"
    fps: int = 24
    video_codec: str = "libx264"
    modalities: str = "frames"
    decode: bool = True
    variant: str | None = None
    ckpt_path: str | None = None
    profile: str | None = None
    set_overrides: Annotated[
        str | None,
        tyro.conf.arg(
            name="set",
            help="Comma-separated runtime overrides (e.g. teacache=off,attention_backend=sdpa).",
        ),
    ] = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    prompt: str | None = None
    no_fetch: bool = False
    quiet: Annotated[bool, tyro.conf.arg(aliases=("-q",))] = False

    def run(self) -> None:
        from worldkernels.cli.run import run_session

        run_session(
            model=self.model,
            steps=self.steps,
            action_type=self.action_type,
            height=self.height,
            width=self.width,
            device=self.device,
            seed=self.seed,
            output_dir=self.output_dir,
            output_format=self.output_format,
            fps=self.fps,
            video_codec=self.video_codec,
            modalities=self.modalities,
            decode=self.decode,
            prompt=self.prompt,
            variant=self.variant,
            ckpt_path=self.ckpt_path,
            model_kwargs=_extra_kwargs(self.num_inference_steps, self.guidance_scale),
            allow_fetch=not self.no_fetch,
            quiet=self.quiet,
            profile=self.profile,
            overrides=_parse_set(self.set_overrides),
        )


@dataclass
class Pull:
    r"""Pre-fetch a model: install deps, clone packages, download weights. No server."""

    model: _PositionalModel = "dummy"
    variant: str | None = None
    ckpt_path: str | None = None
    quiet: Annotated[bool, tyro.conf.arg(aliases=("-q",))] = False

    def run(self) -> None:
        from worldkernels.cli.pull import run_pull

        run_pull(self.model, variant=self.variant, ckpt_path=self.ckpt_path, quiet=self.quiet)


@dataclass
class Models:
    r"""List models: locally cached by default, or the full hub with ``--all``."""

    all: bool = False

    def run(self) -> None:
        from worldkernels.cli.pull import run_models

        run_models(show_all=self.all)


@dataclass
class Rm:
    r"""Remove a model from the local cache (weights + manifest)."""

    model: _PositionalModel = ""
    variant: str | None = None

    def run(self) -> None:
        if not self.model:
            print("Error: model name is required")
            raise SystemExit(1)
        from worldkernels.cli.pull import run_rm

        run_rm(self.model, variant=self.variant)


@dataclass
class CollectEnv:
    r"""Collect environment info: GPU, deps, plugins, hub, local cache, isolated envs."""

    def run(self) -> None:
        from worldkernels.cli.collect_env import run_collect_env

        run_collect_env()


@dataclass
class ConfigShow:
    r"""Show the resolved runtime config (component toggles) and each flag's source."""

    profile: str | None = None
    json: bool = False

    def run(self) -> None:
        from worldkernels.cli.config_cmd import run_config_show

        run_config_show(self.profile, self.json)


@dataclass
class ModelInspect:
    r"""Show model metadata: transition mode, VRAM estimate, capabilities."""

    model: _PositionalModel = "dummy"
    device: str = "cpu"
    config_json: str | None = None

    def run(self) -> None:
        from worldkernels.cli.model import run_inspect

        run_inspect(self.model, self.device, self.config_json)


@dataclass
class BenchLatency:
    r"""Measure single-session step latency."""

    world: str = "dummy"
    steps: int = 100
    height: int = 64
    width: int = 64
    device: str = "cpu"
    profile: str | None = None

    def run(self) -> None:
        from worldkernels.cli.bench import run_latency

        run_latency(
            self.world, self.steps, self.height, self.width, self.device, profile=self.profile
        )


@dataclass
class BenchThroughput:
    r"""Measure multi-session concurrent throughput."""

    world: str = "dummy"
    sessions: int = 4
    steps: int = 50
    height: int = 64
    width: int = 64
    device: str = "cpu"
    profile: str | None = None

    def run(self) -> None:
        from worldkernels.cli.bench import run_throughput

        run_throughput(
            self.world, self.sessions, self.steps, self.height, self.width, self.device,
            profile=self.profile,
        )


@dataclass
class BenchStartup:
    r"""Measure model load + warmup time."""

    world: str = "dummy"
    device: str = "cpu"

    def run(self) -> None:
        from worldkernels.cli.bench import run_startup

        run_startup(self.world, self.device)


@dataclass
class BenchVRAM:
    r"""Profile VRAM estimates across resolutions."""

    world: str = "dummy"
    device: str = "cuda"
    resolutions: str = "256x256,480x848,720x1280"

    def run(self) -> None:
        from worldkernels.cli.bench import run_vram

        run_vram(self.world, self.device, self.resolutions)


@dataclass
class BenchProfile:
    r"""Run steps under torch.profiler and emit a trace file."""

    world: str = "dummy"
    steps: int = 10
    height: int = 64
    width: int = 64
    device: str = "cuda"
    output: str = "wk_profile"

    def run(self) -> None:
        from worldkernels.cli.bench import run_profile

        run_profile(self.world, self.steps, self.height, self.width, self.device, self.output)


@dataclass
class Plugins:
    r"""List discovered entry_point plugins."""

    def run(self) -> None:
        from worldkernels.cli.plugins import run_list

        run_list()


BenchCommand = Union[
    Annotated[BenchLatency, tyro.conf.subcommand("latency")],
    Annotated[BenchThroughput, tyro.conf.subcommand("throughput")],
    Annotated[BenchStartup, tyro.conf.subcommand("startup")],
    Annotated[BenchVRAM, tyro.conf.subcommand("vram")],
    Annotated[BenchProfile, tyro.conf.subcommand("profile")],
]


@dataclass
class Bench:
    r"""Benchmarks: latency, throughput, startup, vram, profile."""

    cmd: BenchCommand

    def run(self) -> None:
        self.cmd.run()


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _parse_set(value: str | None) -> dict | None:
    r"""Parse a ``--set k=v,k=v`` string into a RuntimeConfig overrides dict.

    Bools accept ``on``/``off``/``true``/``false``/``1``/``0``; enums pass through
    as strings. Validation happens in ``resolve_runtime_config``.
    """
    if not value:
        return None
    out: dict = {}
    for pair in value.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError(f"--set entry must be key=value, got {pair!r}")
        key, val = pair.split("=", 1)
        key, val = key.strip(), val.strip()
        lv = val.lower()
        if lv in _BOOL_TRUE:
            out[key] = True
        elif lv in _BOOL_FALSE:
            out[key] = False
        else:
            out[key] = val
    return out


def _extra_kwargs(num_inference_steps: int | None, guidance_scale: float | None) -> dict:
    out: dict = {}
    if num_inference_steps is not None:
        out["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        out["guidance_scale"] = guidance_scale
    return out


Command = tyro.conf.SuppressFixed[
    Union[
        Annotated[Serve, tyro.conf.subcommand("serve")],
        Annotated[Run, tyro.conf.subcommand("run")],
        Annotated[Pull, tyro.conf.subcommand("pull")],
        Annotated[Models, tyro.conf.subcommand("models")],
        Annotated[Rm, tyro.conf.subcommand("rm")],
        Annotated[CollectEnv, tyro.conf.subcommand("collect-env")],
        Annotated[ConfigShow, tyro.conf.subcommand("config-show")],
        Annotated[ModelInspect, tyro.conf.subcommand("inspect")],
        Annotated[Bench, tyro.conf.subcommand("bench")],
        Annotated[Plugins, tyro.conf.subcommand("plugins")],
    ]
]


def app() -> None:
    if "--version" in sys.argv or "-V" in sys.argv:
        from worldkernels import __version__

        print(f"worldkernels {__version__}")
        raise SystemExit(0)
    cmd = tyro.cli(Command, description="worldkernels — GPU-first world model simulation engine")  # type: ignore[call-overload]
    cmd.run()


if __name__ == "__main__":
    app()
