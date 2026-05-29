r"""Benchmark command implementations."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from worldkernels import Action, WorldConfig, WorldEngine
from worldkernels.core.session import Session


@contextmanager
def bench_env(
    world: str,
    device: str,
    max_sessions: int = 1,
    height: int = 64,
    width: int = 64,
    num_sessions: int = 1,
    profile: str | None = None,
) -> Generator[tuple[WorldEngine, list[Session]], None, None]:
    r"""Shared setup/teardown for benchmark commands."""
    if profile is not None:
        wk = WorldEngine(profile, device=device, max_sessions=max_sessions)
    else:
        wk = WorldEngine(device=device, max_sessions=max_sessions)
    wk.load_model(world)
    world_key = world.split("/")[-1]
    config = WorldConfig(height=height, width=width, frames_per_step=1)
    sessions = [wk.create_session(world_key, config=config, seed=i) for i in range(num_sessions)]
    try:
        yield wk, sessions
    finally:
        for sess in sessions:
            sess.close()
        wk.shutdown()


def run_latency(
    world: str, steps: int, height: int, width: int, device: str, profile: str | None = None
) -> None:
    with bench_env(world, device, height=height, width=width, profile=profile) as (_, sessions):
        session = sessions[0]
        latencies: list[float] = []
        for _ in range(steps):
            t0 = time.perf_counter()
            session.step(Action("null", {}), modalities=["frames"])
            latencies.append((time.perf_counter() - t0) * 1000.0)

        latencies.sort()
        n = len(latencies)
        print(f"world={world}  steps={n}  device={device}")
        print(f"  mean:  {sum(latencies) / n:.2f} ms")
        print(f"  p50:   {latencies[n // 2]:.2f} ms")
        print(f"  p99:   {latencies[int(n * 0.99)]:.2f} ms")
        print(f"  min:   {latencies[0]:.2f} ms")
        print(f"  max:   {latencies[-1]:.2f} ms")


def run_throughput(
    world: str,
    num_sessions: int,
    steps: int,
    height: int,
    width: int,
    device: str,
    profile: str | None = None,
) -> None:
    with bench_env(
        world,
        device,
        max_sessions=num_sessions,
        height=height,
        width=width,
        num_sessions=num_sessions,
        profile=profile,
    ) as (_, sessions):
        t0 = time.perf_counter()
        for _ in range(steps):
            for sess in sessions:
                sess.step(Action("null", {}), modalities=["frames"])
        elapsed = time.perf_counter() - t0

        total_steps = steps * num_sessions
        print(f"world={world}  sessions={num_sessions}  device={device}")
        print(f"  total steps: {total_steps}")
        print(f"  wall time:   {elapsed:.2f} s")
        print(f"  throughput:  {total_steps / elapsed:.1f} steps/s")


def run_vram(
    world: str,
    device: str = "cuda",
    resolutions: str = "256x256,480x848,720x1280",
) -> None:
    r"""Profile VRAM across resolutions."""
    import torch

    from worldkernels.core.config import WorldConfig
    from worldkernels.worlds.hub import resolve_model
    from worldkernels.worlds.registry import get_world_class

    adapter_name, merged_kwargs = resolve_model(world)
    cls = get_world_class(adapter_name)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    instance = cls(**merged_kwargs)
    instance.initialize(device=device, dtype=dtype)

    pairs = []
    for res in resolutions.split(","):
        parts = res.strip().split("x")
        pairs.append((int(parts[0]), int(parts[1])))

    print(f"world={world}  device={device}")
    print(f"{'Resolution':>14s}  {'VRAM (MB)':>10s}")
    print("-" * 28)
    for h, w in pairs:
        cfg = WorldConfig(height=h, width=w)
        vram = instance.profile_vram(cfg)
        print(f"  {h:4d}x{w:<4d}      {vram:8.1f}")


def run_startup(world: str, device: str) -> None:
    t0 = time.perf_counter()
    wk = WorldEngine(device=device)
    t_engine = time.perf_counter() - t0

    t1 = time.perf_counter()
    wk.load_model(world)
    t_load = time.perf_counter() - t1

    print(f"world={world}  device={device}")
    print(f"  engine init: {t_engine * 1000:.1f} ms")
    print(f"  load+warmup: {t_load * 1000:.1f} ms")
    print(f"  total:       {(t_engine + t_load) * 1000:.1f} ms")

    wk.shutdown()


def run_profile(
    world: str,
    steps: int = 10,
    height: int = 64,
    width: int = 64,
    device: str = "cuda",
    output: str = "wk_profile",
) -> None:
    r"""Run steps under torch.profiler and emit a Chrome/Nsight trace."""
    from pathlib import Path

    import torch

    with bench_env(world, device, height=height, width=width) as (_, sessions):
        session = sessions[0]

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                *(
                    [torch.profiler.ProfilerActivity.CUDA]
                    if device != "cpu" and torch.cuda.is_available()
                    else []
                ),
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(steps):
                session.step(Action("null", {}), modalities=["frames"])

    trace_path = Path(f"{output}.json")
    prof.export_chrome_trace(str(trace_path))
    print(f"Trace written to {trace_path}")
    print("Open in chrome://tracing or Perfetto UI")

    print()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
