r"""Marker test for tests/fixtures/__init__.py."""


def test_module_importable():
    import tests.fixtures as f

    assert f is not None
