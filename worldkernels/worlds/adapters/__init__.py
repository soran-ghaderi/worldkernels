r"""World model adapters.

Each adapter lives in its own subpackage with a standard layout::

    adapters/<adapter_name>/
        __init__.py    # lazy public exports via __getattr__
        adapter.py     # AbstractWorld subclass

Adapters sharing a model family use a private family package::

    adapters/_<family_name>/
        __init__.py    # lazy re-exports of shared internals
        _base.py       # shared base class
        _deps.py       # dependency setup

Adapters are loaded lazily via the registry. Direct imports are also supported:

    from worldkernels.worlds.adapters.dummy import DummyWorld
    from worldkernels.worlds.adapters.dreamdojo import DreamDojoWorld
"""
