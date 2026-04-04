r"""WorldKernels CLI.

Pure schema: each dataclass defines a subcommand's flags.
Implementations live in sibling modules.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Annotated, Union

import tyro

# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@dataclass
class Serve:
    r"""Start the WorldKernels HTTP/WebSocket server."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_sessions: int = 4
    api_key: Annotated[str | None, tyro.conf.arg(aliases=("-k",))] = None
    device: str = "cuda"
    model: str | None = None
    ckpt_path: str | None = None
    experiment: str | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None

    def run(self) -> None:
        from worldkernels.cli.serve import run_serve

        model_kwargs: dict = {}
        if self.ckpt_path is not None:
            model_kwargs["ckpt_path"] = self.ckpt_path
        if self.experiment is not None:
            model_kwargs["experiment"] = self.experiment
        if self.num_inference_steps is not None:
            model_kwargs["num_inference_steps"] = self.num_inference_steps
        if self.guidance_scale is not None:
            model_kwargs["guidance_scale"] = self.guidance_scale

        run_serve(
            self.host,
            self.port,
            self.max_sessions,
            self.api_key,
            self.device,
            self.model,
            model_kwargs,
        )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@dataclass
class Run:
    r"""Run a headless session: load model, step N times, optionally save frames."""

    world: str = "dummy"
    steps: int = 10
    action_type: str = "null"
    height: int = 480
    width: int = 848
    device: str = "cuda"
    seed: int = 0
    output_dir: str | None = None
    output_format: str = "frames"
    fps: int = 24
    video_codec: str = "libx264"
    modalities: str = "frames"
    decode: bool = True
    ckpt_path: str | None = None
    experiment: str | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    prompt: str | None = None

    def run(self) -> None:
        from worldkernels.cli.run import run_session

        model_kwargs: dict = {}
        if self.ckpt_path is not None:
            model_kwargs["ckpt_path"] = self.ckpt_path
        if self.experiment is not None:
            model_kwargs["experiment"] = self.experiment
        if self.num_inference_steps is not None:
            model_kwargs["num_inference_steps"] = self.num_inference_steps
        if self.guidance_scale is not None:
            model_kwargs["guidance_scale"] = self.guidance_scale

        run_session(
            self.world,
            self.steps,
            self.action_type,
            self.height,
            self.width,
            self.device,
            self.seed,
            self.output_dir,
            self.output_format,
            self.fps,
            self.video_codec,
            self.modalities,
            self.decode,
            self.prompt,
            model_kwargs,
        )


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@dataclass
class Doctor:
    r"""Check system environment: GPU, dependencies, plugins, world models."""

    def run(self) -> None:
        from worldkernels.cli.doctor import run_doctor

        run_doctor()


# ---------------------------------------------------------------------------
# model:*
# ---------------------------------------------------------------------------


@dataclass
class ModelList:
    r"""List registered world models."""

    verbose: Annotated[bool, tyro.conf.arg(aliases=("-v",))] = False

    def run(self) -> None:
        from worldkernels.cli.model import run_list

        run_list(self.verbose)


@dataclass
class ModelInspect:
    r"""Show model metadata: transition mode, VRAM estimate, capabilities."""

    model_id: str = "dummy"
    device: str = "cpu"
    config_json: str | None = None

    def run(self) -> None:
        from worldkernels.cli.model import run_inspect

        run_inspect(self.model_id, self.device, self.config_json)


@dataclass
class ModelDownload:
    r"""Download a model from HuggingFace Hub."""

    model_id: str = ""
    revision: str | None = None
    cache_dir: str | None = None

    def run(self) -> None:
        if not self.model_id:
            print("Error: --model-id is required")
            raise SystemExit(1)
        from worldkernels.cli.model import run_download

        run_download(self.model_id, self.revision, self.cache_dir)


@dataclass
class ModelRemove:
    r"""Remove a model from the HuggingFace cache."""

    model_id: str = ""

    def run(self) -> None:
        if not self.model_id:
            print("Error: --model-id is required")
            raise SystemExit(1)
        from worldkernels.cli.model import run_remove

        run_remove(self.model_id)


@dataclass
class ModelExport:
    r"""Export a model to TensorRT or ONNX format."""

    model_id: str = "dummy"
    fmt: str = "tensorrt"
    output: str | None = None
    height: int = 480
    width: int = 848
    device: str = "cuda"

    def run(self) -> None:
        from worldkernels.cli.model import run_export

        run_export(self.model_id, self.fmt, self.output, self.height, self.width, self.device)


# ---------------------------------------------------------------------------
# bench:*
# ---------------------------------------------------------------------------


@dataclass
class BenchLatency:
    r"""Measure single-session step latency."""

    world: str = "dummy"
    steps: int = 100
    height: int = 64
    width: int = 64
    device: str = "cpu"

    def run(self) -> None:
        from worldkernels.cli.bench import run_latency

        run_latency(self.world, self.steps, self.height, self.width, self.device)


@dataclass
class BenchThroughput:
    r"""Measure multi-session concurrent throughput."""

    world: str = "dummy"
    sessions: int = 4
    steps: int = 50
    height: int = 64
    width: int = 64
    device: str = "cpu"

    def run(self) -> None:
        from worldkernels.cli.bench import run_throughput

        run_throughput(self.world, self.sessions, self.steps, self.height, self.width, self.device)


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


# ---------------------------------------------------------------------------
# plugin:*
# ---------------------------------------------------------------------------


@dataclass
class PluginList:
    r"""List discovered entry_point plugins."""

    def run(self) -> None:
        from worldkernels.cli.plugin import run_list

        run_list()


# ---------------------------------------------------------------------------
# Command union
# ---------------------------------------------------------------------------


Command = tyro.conf.SuppressFixed[
    Union[
        Annotated[Serve, tyro.conf.subcommand("serve")],
        Annotated[Run, tyro.conf.subcommand("run")],
        Annotated[Doctor, tyro.conf.subcommand("doctor")],
        Annotated[ModelList, tyro.conf.subcommand("model:list", prefix_name=False)],
        Annotated[ModelInspect, tyro.conf.subcommand("model:inspect", prefix_name=False)],
        Annotated[ModelDownload, tyro.conf.subcommand("model:download", prefix_name=False)],
        Annotated[ModelRemove, tyro.conf.subcommand("model:remove", prefix_name=False)],
        Annotated[ModelExport, tyro.conf.subcommand("model:export", prefix_name=False)],
        Annotated[BenchLatency, tyro.conf.subcommand("bench:latency", prefix_name=False)],
        Annotated[BenchThroughput, tyro.conf.subcommand("bench:throughput", prefix_name=False)],
        Annotated[BenchStartup, tyro.conf.subcommand("bench:startup", prefix_name=False)],
        Annotated[BenchVRAM, tyro.conf.subcommand("bench:vram", prefix_name=False)],
        Annotated[BenchProfile, tyro.conf.subcommand("bench:profile", prefix_name=False)],
        Annotated[PluginList, tyro.conf.subcommand("plugin:list", prefix_name=False)],
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
