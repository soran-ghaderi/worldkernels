r"""Tests for worldkernels/serving/server.py."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from worldkernels.core.config import ServerConfig
from worldkernels.serving.server import create_app


@pytest.fixture
def app():
    cfg = ServerConfig(max_sessions=2, api_key=None)
    a = create_app(cfg, device="cpu")
    yield a
    a.state.engine.shutdown()


@pytest.fixture
def app_with_auth():
    cfg = ServerConfig(max_sessions=2, api_key="secret-key")
    a = create_app(cfg, device="cpu")
    yield a
    a.state.engine.shutdown()


class TestCreateApp:
    def test_returns_fastapi_instance(self, app):
        assert isinstance(app, FastAPI)
        assert app.title == "WorldKernels"
        assert app.version == "0.1.0"

    def test_engine_attached_to_state(self, app):
        from worldkernels.engine import WorldEngine

        assert isinstance(app.state.engine, WorldEngine)

    def test_default_config_used_when_none(self):
        a = create_app(None, device="cpu")
        try:
            assert a.state.engine.max_sessions == 4
        finally:
            a.state.engine.shutdown()


class TestHealthEndpoint:
    def test_returns_ok(self, app):
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestAuth:
    def test_health_unprotected(self, app_with_auth):
        r = TestClient(app_with_auth).get("/health")
        assert r.status_code == 200

    def test_protected_route_requires_key(self, app_with_auth):
        r = TestClient(app_with_auth).get("/v1/worlds")
        assert r.status_code == 401

    def test_protected_route_with_key(self, app_with_auth):
        client = TestClient(app_with_auth)
        r = client.get("/v1/worlds", headers={"Authorization": "Bearer secret-key"})
        assert r.status_code == 200


class TestShutdownHook:
    def test_shutdown_clears_engine(self):
        cfg = ServerConfig(max_sessions=2)
        a = create_app(cfg, device="cpu")
        a.state.engine.load_model("dummy")
        with TestClient(a):
            pass
        assert a.state.engine._worlds == {}
