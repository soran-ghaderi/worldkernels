r"""World model adapters.

Each adapter lives in its own subpackage with a standard layout::

    adapters/<model_name>/
        __init__.py    # public exports
        adapter.py     # AbstractWorld subclass
        state.py       # model-specific LatentState container

Adapters are loaded lazily via the registry. Direct imports are also supported:

    from worldkernels.worlds.adapters.dummy import DummyWorld
    from worldkernels.worlds.adapters.dreamdojo import DreamDojoWorld
"""
