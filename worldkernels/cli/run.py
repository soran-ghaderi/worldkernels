r"""Headless session runner — load model, step N times, optionally save frames/video."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_session(
    world: str,
    steps: int = 10,
    action_type: str = "null",
    height: int = 480,
    width: int = 848,
    device: str = "cuda",
    seed: int = 0,
    output_dir: str | None = None,
    output_format: str = "frames",
    fps: int = 24,
    video_codec: str = "libx264",
    modalities: str = "frames",
    decode: bool = True,
    prompt: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> None:
    from worldkernels import Action, WorldConfig, WorldEngine

    valid_formats = ("frames", "video", "both")
    if output_format not in valid_formats:
        raise ValueError(f"--output-format must be one of {valid_formats}, got '{output_format}'")

    wk = WorldEngine(device=device)
    world_key = world.split("/")[-1]
    wk.load_model(world, **(model_kwargs or {}))

    mods = [m.strip() for m in modalities.split(",")]
    cfg = WorldConfig(height=height, width=width, initial_prompt=prompt or "")
    session = wk.create_session(world_key, config=cfg, seed=seed)

    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    save_frames = output_format in ("frames", "both")
    save_video = output_format in ("video", "both")

    print(f"Running {steps} steps on '{world}' ({height}x{width}, device={device})")

    frame_counter = 0
    collected_frames: list[Any] = []

    for i in range(steps):
        obs = session.step(
            Action(action_type, {}),
            modalities=mods,
            decode=decode,
        )
        print(f"  step {i:4d}  time={obs.generation_time_ms:.1f}ms", end="")

        if out and obs.frames is not None:
            if save_frames:
                saved = _save_frames(obs.frames, out, height, width, frame_counter)
                frame_counter += saved
                print(f"  -> saved {saved} frames", end="")
            if save_video:
                collected_frames.extend(_raw_to_arrays(obs.frames, height, width))
        print()

    if out and save_video and collected_frames:
        video_path = out / "output.mp4"
        _save_video(collected_frames, video_path, fps, video_codec)
        print(f"Video saved to {video_path} ({len(collected_frames)} frames, {fps} fps)")

    session.close()
    wk.shutdown()
    print(f"Done. {steps} steps, {frame_counter} frames saved to {out or '(no output dir)'}.")


def _save_frames(frames: Any, out_dir: Path, height: int, width: int, start_idx: int) -> int:
    from PIL import Image

    saved = 0
    if isinstance(frames, list):
        for j, raw in enumerate(frames):
            path = out_dir / f"frame_{start_idx + j:05d}.png"
            if isinstance(raw, (bytes, bytearray)):
                img = Image.frombytes("RGB", (width, height), raw)
                img.save(path)
                saved += 1
    return saved


def _raw_to_arrays(frames: Any, height: int, width: int) -> list:
    import numpy as np

    arrays = []
    if isinstance(frames, list):
        for raw in frames:
            if isinstance(raw, (bytes, bytearray)):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                arrays.append(arr)
    return arrays


def _save_video(frames: list, path: Path, fps: int, codec: str) -> None:
    import imageio.v3 as iio
    import numpy as np

    stack = np.stack(frames)
    iio.imwrite(str(path), stack, fps=fps, codec=codec)
