from worldkernels import Action, WorldConfig, WorldKernel


def main() -> None:
    wk = WorldKernel(device="cpu")
    wk.load_world("dummy")
    session = wk.create_session("dummy", config=WorldConfig(height=64, width=64), seed=7)
    observation = session.step(Action("keyboard", {"keys": ["W"]}), modalities=["frames"])
    print("step:", observation.step_index, "frames:", len(observation.frames or []))
    session.close()
    wk.shutdown()


if __name__ == "__main__":
    main()
