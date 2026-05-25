r"""Per-test reload of the routes module.

`worldkernels.serving.routes` defines a single module-level ``APIRouter``
that ``configure_routes`` appends to. Across tests this accumulates duplicate
routes whose Depends-bound engines point at the first app. Reloading the
module per test gives every app a clean router."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _fresh_routes_module():
    import worldkernels.serving.routes
    import worldkernels.serving.server

    importlib.reload(worldkernels.serving.routes)
    importlib.reload(worldkernels.serving.server)
    yield
