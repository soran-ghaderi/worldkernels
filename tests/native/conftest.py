r"""Pytest plumbing for native-rewrite tests: --regen-fixtures and reference_dir."""

from __future__ import annotations

from pathlib import Path

import pytest

_FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "data"


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("worldkernels-fixtures")
    g.addoption(
        "--regen-fixtures",
        action="store_true",
        default=False,
        help="Re-capture cosmos_predict2 golden tensors before running tests. "
        "Requires GPU + cosmos_predict2 install. See tests/fixtures/cosmos_reference.py.",
    )
    g.addoption(
        "--fixture-adapter",
        default="dreamdojo",
        choices=("dreamdojo", "cosmos"),
        help="Which adapter to capture against.",
    )
    g.addoption("--fixture-variant", default="2b_pretrain")
    g.addoption(
        "--fixture-spatial",
        default="240x320",
        help="HxW for captured fixtures (e.g. 240x320).",
    )
    g.addoption("--fixture-frames", type=int, default=5)
    g.addoption("--fixture-dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    g.addoption("--fixture-seed", type=int, default=1234)
    g.addoption("--fixture-steps", type=int, default=4)


def _parse_spatial(s: str) -> tuple[int, int]:
    h, w = s.lower().split("x", 1)
    return int(h), int(w)


@pytest.fixture(scope="session")
def fixture_cfg(request: pytest.FixtureRequest):
    from tests.fixtures.cosmos_reference import CaptureConfig

    h, w = _parse_spatial(request.config.getoption("--fixture-spatial"))
    return CaptureConfig(
        adapter=request.config.getoption("--fixture-adapter"),
        variant=request.config.getoption("--fixture-variant"),
        height=h,
        width=w,
        pixel_frames=request.config.getoption("--fixture-frames"),
        dtype_str=request.config.getoption("--fixture-dtype"),
        seed=request.config.getoption("--fixture-seed"),
        num_steps=request.config.getoption("--fixture-steps"),
        output_root=_FIXTURE_ROOT,
    )


@pytest.fixture(scope="session")
def reference_dir(request: pytest.FixtureRequest, fixture_cfg) -> Path:
    r"""Path to the active fixture directory.

    If `--regen-fixtures` is passed, capture is invoked before the directory is
    returned. Otherwise the directory must already exist (committed locally or
    downloaded from the worldkernels HF dataset); tests that depend on it are
    skipped when it is missing.
    """
    out_dir = fixture_cfg.output_dir
    if request.config.getoption("--regen-fixtures"):
        from tests.fixtures.cosmos_reference import capture

        capture(fixture_cfg)
    if not out_dir.exists():
        pytest.skip(
            f"fixture dir {out_dir} not found; run with --regen-fixtures "
            "or download the worldkernels reference dataset"
        )
    return out_dir
