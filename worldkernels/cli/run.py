r"""Headless session runner — load model, step N times, optionally save frames."""

from __future__ import annotations

from pathlib import Path


def run_session(
    world: str,
    steps: int = 10,
    action_type: str = "null",
    height: int = 480,
    width: int = 848,
    device: str = "cuda",
    seed: int = 0,
    output_dir: str | None = None,
    modalities: str = "frames",
    decode: bool = True,
) -> None:
    import torch
    from worldkernels import Action, WorldConfig, WorldKernel

    wk = WorldKernel(device=device)
    wk.load_model(world)

    mods = [m.strip() for m in modalities.split(",")]
    cfg = WorldConfig(height=height, width=width)
    session = wk.create_session(world, config=cfg, seed=seed)

    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    print(f"Running {steps} steps on '{world}' ({height}x{width}, device={device})")

    for i in range(steps):
        obs = session.step(
            Action(action_type, {}),
            modalities=mods,
            decode=decode,
        )
        print(f"  step {i:4d}  time={obs.generation_time_ms:.1f}ms", end="")

        if out and obs.frames is not None:
            frame_path = out / f"frame_{i:05d}.pt"
            if isinstance(obs.frames, torch.Tensor):
                torch.save(obs.frames.cpu(), frame_path)
                print(f"  -> {frame_path}", end="")
        print()

    session.close()
    wk.shutdown()
    print(f"Done. {steps} steps completed.")
