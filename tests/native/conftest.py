r"""Pytest plumbing for native-regression tests: --regen-fixtures and reference_dir."""

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
        help="Re-capture DreamDojo golden tensors before running tests (GPU + weights).",
    )
    g.addoption("--fixture-variant", default="2b_pretrain")
    g.addoption("--fixture-spatial", default="240x320", help="HxW, e.g. 240x320.")
    g.addoption("--fixture-dtype", default="bfloat16", choices=("bfloat16", "float32"))
    g.addoption("--fixture-seed", type=int, default=1234)
    g.addoption("--fixture-steps", type=int, default=4)
    g.addoption("--fixture-guidance", type=float, default=0.0)


def _parse_spatial(s: str) -> tuple[int, int]:
    h, w = s.lower().split("x", 1)
    return int(h), int(w)


@pytest.fixture(scope="session")
def fixture_cfg(request: pytest.FixtureRequest):
    from tests.fixtures.dreamdojo_reference import CaptureConfig

    h, w = _parse_spatial(request.config.getoption("--fixture-spatial"))
    return CaptureConfig(
        variant=request.config.getoption("--fixture-variant"),
        height=h,
        width=w,
        dtype_str=request.config.getoption("--fixture-dtype"),
        seed=request.config.getoption("--fixture-seed"),
        num_steps=request.config.getoption("--fixture-steps"),
        guidance=request.config.getoption("--fixture-guidance"),
        output_root=_FIXTURE_ROOT,
    )


@pytest.fixture(scope="session")
def reference_dir(request: pytest.FixtureRequest, fixture_cfg) -> Path:
    r"""Path to the active fixture directory.

    With ``--regen-fixtures`` the capture runs first; otherwise the directory
    must already exist and dependent tests skip when it is missing.
    """
    out_dir = fixture_cfg.output_dir
    if request.config.getoption("--regen-fixtures"):
        from tests.fixtures.dreamdojo_reference import capture

        capture(fixture_cfg)
    if not out_dir.exists():
        pytest.skip(
            f"fixture dir {out_dir} not found; run with --regen-fixtures "
            "or download the worldkernels reference dataset"
        )
    return out_dir
