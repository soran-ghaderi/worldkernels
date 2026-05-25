r"""Tests for worldkernels/plugins.py (entry-point system; currently doc-only)."""


def test_module_importable():
    import worldkernels.plugins as plugins

    assert plugins is not None
    assert plugins.__doc__ is not None
