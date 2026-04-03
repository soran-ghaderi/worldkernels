"""End-to-end example: create a DummyWorld session and step through it.

Run with:
    python examples/dummy_session.py

No GPU required. Works on CPU.
"""

from worldkernels import Action, WorldConfig, WorldKernel


def main() -> None:
    # 1. Create the engine (CPU for laptop dev)
    wk = WorldKernel(device="cpu")

    # 2. Load the built-in dummy world
    wk.load_model("dummy")
    print(f"Loaded worlds: {wk.list_worlds()}")

    # 3. Create a session with small resolution for fast iteration
    config = WorldConfig(height=64, width=64, frames_per_step=1)
    session = wk.create_session("dummy", config=config, seed=42)
    print(f"Session: {session.id}, status={session.status.value}")

    # 4. Take a few steps
    for i in range(5):
        action = Action("keyboard", {"keys": ["W"]})
        obs = session.step(action, modalities=["frames"])
        frame_size = len(obs.frames[0]) if obs.frames else 0
        print(
            f"  step {session.step_index}: "
            f"gen_time={obs.generation_time_ms:.2f}ms, "
            f"frame_bytes={frame_size}"
        )

    # 5. Checkpoint and branch
    ckpt_id = session.checkpoint()
    print(f"\nCheckpoint saved: {ckpt_id}")

    branched = session.branch()
    print(f"Branched session: {branched.id} (parent={branched.parent_session_id})")

    # Take a different action in the branch
    obs_branch = branched.step(
        Action("keyboard", {"keys": ["A"]}),
        modalities=["frames", "depth"],
    )
    print(
        f"  branch step {branched.step_index}: "
        f"gen_time={obs_branch.generation_time_ms:.2f}ms, "
        f"has_depth={obs_branch.depth is not None}"
    )

    # Original session continues independently
    obs_orig = session.step(
        Action("keyboard", {"keys": ["D"]}),
        modalities=["frames"],
    )
    print(f"  original step {session.step_index}: gen_time={obs_orig.generation_time_ms:.2f}ms")

    # 6. Restore to checkpoint
    session.restore(ckpt_id)
    print(f"\nRestored to {ckpt_id}, step_index still={session.step_index}")

    # 7. Cleanup
    session.close()
    branched.close()
    wk.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
