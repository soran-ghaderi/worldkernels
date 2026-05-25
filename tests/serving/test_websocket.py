r"""Tests for the WebSocket session-streaming route (CPU-safe)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from worldkernels.config import ServerConfig
from worldkernels.serving.server import create_app


@pytest.fixture
def client():
    app = create_app(ServerConfig(max_sessions=4, api_key=None), device="cpu")
    app.state.engine.load_model("dummy")
    with TestClient(app) as c:
        yield c
    app.state.engine.shutdown()


def _session_id(client: TestClient) -> str:
    body = client.post(
        "/v1/sessions",
        json={"world": "dummy", "height": 32, "width": 32, "frames_per_step": 1},
    ).json()
    return body["session_id"]


class TestWebSocketStream:
    def test_streams_observations(self, client):
        session_id = _session_id(client)
        with client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            ws.send_json({"action_type": "null", "modalities": ["frames"]})
            obs = ws.receive_json()
            assert obs["step_index"] >= 0
            assert "frames" in obs

    def test_multiple_steps_over_one_connection(self, client):
        session_id = _session_id(client)
        with client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
            for _ in range(3):
                ws.send_json({"action_type": "null"})
                ws.receive_json()

    def test_unknown_session_reports_error(self, client):
        with client.websocket_connect("/v1/sessions/ghost/stream") as ws:
            assert "error" in ws.receive_json()
