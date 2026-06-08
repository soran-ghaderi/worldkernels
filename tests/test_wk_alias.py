r"""Tests for the `wk` import alias: identity preservation and lazy-import safety."""

from __future__ import annotations

import subprocess
import sys

import worldkernels


def test_import_wk_stays_torch_free():
    code = (
        "import sys, wk; "
        "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.')); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_submodule_identity():
    import wk.engine
    import worldkernels.engine

    assert wk.engine is worldkernels.engine


def test_deep_import_identity():
    from wk.engine import WorldEngine as ViaAlias
    from worldkernels.engine import WorldEngine as ViaReal

    assert ViaAlias is ViaReal


def test_top_level_attr_delegation():
    import wk

    assert wk.WorldEngine is worldkernels.WorldEngine


def test_from_wk_import():
    from wk import Action as ViaAlias

    assert ViaAlias is worldkernels.Action
