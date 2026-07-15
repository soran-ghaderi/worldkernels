r"""Conditioning DreamDojo: init image and action chunks (joints or latent actions).

DreamDojo is an action-conditioned image-to-video world model. Its conditioning
signals are wired through the public API as shown below. Run with:

    python examples/dreamdojo_conditioning.py --image init.png --steps 3

Requires a CUDA GPU and downloads the 2B checkpoint + Wan2.1 VAE on first run.
"""

from __future__ import annotations

import argparse

import numpy as np

from worldkernels import Action, WorldConfig, WorldEngine

ACTION_DIM = 384  # DreamDojo robot joint vector size
CHUNK_SIZE = 12  # frames per transition the action conditions


def synthetic_action(step: int) -> Action:
    r"""A reproducible synthetic joint trajectory of shape (CHUNK_SIZE, ACTION_DIM).

    Real use: replace ``joints`` with your robot's per-frame joint vectors (a teleop log, a
    policy rollout, etc.). A 1-D ``(ACTION_DIM,)`` vector is broadcast across the chunk;
    a 2-D ``(CHUNK_SIZE, ACTION_DIM)`` array gives a distinct action per frame.
    """
    t = np.linspace(0.0, 1.0, CHUNK_SIZE)[:, None]
    phase = 0.5 * step
    joints = 0.6 * np.sin(2 * np.pi * (t + phase) + np.arange(ACTION_DIM)[None, :] * 0.01)
    return Action("joints", {"joints": joints.astype(np.float32).tolist()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="path to the initial frame (anchors the video)")
    ap.add_argument("--variant", default="2b_pretrain")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--latent-actions", action="store_true", help="use 32-dim latent actions")
    args = ap.parse_args()

    wk = WorldEngine(device="cuda")
    wk.load_model("dreamdojo", variant=args.variant)
    world_key = next(iter(wk.list_worlds()))

    cfg = WorldConfig(
        height=args.height,
        width=args.width,
        initial_prompt="a robot arm picking up a red cube on a desk",
        initial_image=args.image,
    )
    session = wk.create_session(world_key, config=cfg, seed=0)

    for i in range(args.steps):
        if args.latent_actions:
            z = (0.5 * np.sin(np.arange(32) * 0.3 + i)).astype(np.float32).tolist()
            action = Action("latent", {"latent_action": z})
        else:
            action = synthetic_action(i)
        obs = session.step(action, modalities=["frames"])
        n = len(obs.frames) if obs.frames else 0
        print(f"step {i}: {n} frames, {obs.generation_time_ms:.0f}ms")

    session.close()
    wk.shutdown()


if __name__ == "__main__":
    main()
