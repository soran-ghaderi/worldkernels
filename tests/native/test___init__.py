r"""Marker test for tests/native/__init__.py."""


def test_module_importable():
    import tests.native as n

    assert n is not None
