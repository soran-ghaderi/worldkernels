r"""Tests for worldkernels/serving/routes.py."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from worldkernels.core.config import ServerConfig
from worldkernels.serving.routes import _observation_to_dict, _session_summary
from worldkernels.serving.server import create_app


@pytest.fixture
def client():
    app = create_app(ServerConfig(max_sessions=4, api_key=None), device="cpu")
    app.state.engine.load_model("dummy")
    with TestClient(app) as c:
        yield c
    app.state.engine.shutdown()


class TestWorldRoutes:
    def test_list_worlds(self, client):
        r = client.get("/v1/worlds")
        assert r.status_code == 200
        assert "dummy" in r.json()["worlds"]

    def test_load_already_loaded_409(self, client):
        r = client.post("/v1/worlds", json={"model_id": "dummy"})
        assert r.status_code == 409

    def test_load_unknown_404(self, client):
        r = client.post("/v1/worlds", json={"model_id": "ghost_xyz"})
        assert r.status_code == 404

    def test_load_with_alias(self, client):
        r = client.post("/v1/worlds", json={"model_id": "dummy", "alias": "alt"})
        assert r.status_code == 201
        assert r.json() == {"status": "loaded", "world": "alt"}

    def test_unload(self, client):
        r = client.delete("/v1/worlds/dummy")
        assert r.status_code == 200
        assert r.json()["status"] == "unloaded"

    def test_unload_unknown_404(self, client):
        r = client.delete("/v1/worlds/missing_xyz")
        assert r.status_code == 404


class TestSessionRoutes:
    def _make_session(self, client):
        r = client.post(
            "/v1/sessions",
            json={"world": "dummy", "height": 64, "width": 64, "frames_per_step": 1},
        )
        assert r.status_code == 201
        return r.json()

    def test_list_sessions_empty(self, client):
        r = client.get("/v1/sessions")
        assert r.json() == {"sessions": []}

    def test_create_session(self, client):
        body = self._make_session(client)
        assert body["world_id"] == "dummy"
        assert body["status"] == "active"

    def test_create_session_unknown_world_404(self, client):
        r = client.post("/v1/sessions", json={"world": "missing"})
        assert r.status_code == 404

    def test_session_limit_429(self):
        app = create_app(ServerConfig(max_sessions=1, api_key=None), device="cpu")
        app.state.engine.load_model("dummy")
        c = TestClient(app)
        try:
            c.post(  # noqa: E501
                "/v1/sessions",  # noqa: E501
                json={"world": "dummy", "height": 32, "width": 32, "frames_per_step": 1},
            )
            r = c.post(
                "/v1/sessions",
                json={"world": "dummy", "height": 32, "width": 32, "frames_per_step": 1},
            )
            assert r.status_code == 429
        finally:
            app.state.engine.shutdown()

    def test_get_session(self, client):
        body = self._make_session(client)
        r = client.get(f"/v1/sessions/{body['session_id']}")
        assert r.status_code == 200
        assert r.json()["session_id"] == body["session_id"]

    def test_get_session_unknown_404(self, client):
        r = client.get("/v1/sessions/missing_xyz")
        assert r.status_code == 404

    def test_delete_session(self, client):
        body = self._make_session(client)
        r = client.delete(f"/v1/sessions/{body['session_id']}")
        assert r.status_code == 200
        assert r.json()["status"] == "terminated"

    def test_delete_session_unknown_404(self, client):
        r = client.delete("/v1/sessions/missing_xyz")
        assert r.status_code == 404

    def test_step(self, client):
        body = self._make_session(client)
        r = client.post(
            f"/v1/sessions/{body['session_id']}/step",
            json={"action_type": "null", "modalities": ["frames"]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == body["session_id"]
        assert "stage_timing" in data
        assert data.get("num_frames") == 1

    def test_step_decode_false(self, client):
        body = self._make_session(client)
        r = client.post(
            f"/v1/sessions/{body['session_id']}/step",
            json={"action_type": "null", "decode": False},
        )
        assert r.status_code == 200
        assert r.json()["decode_skipped"] is True

    def test_step_unknown_session_404(self, client):
        r = client.post("/v1/sessions/ghost/step", json={"action_type": "null"})
        assert r.status_code == 404

    def test_step_on_terminated_410(self, client):
        body = self._make_session(client)
        client.delete(f"/v1/sessions/{body['session_id']}")
        body["session_id"]
        sess = client.app.state.engine.get_session(body["session_id"])
        assert sess is None

    def test_checkpoint_and_restore(self, client):
        body = self._make_session(client)
        client.post(
            f"/v1/sessions/{body['session_id']}/step",
            json={"action_type": "null"},
        )
        ckpt = client.post(f"/v1/sessions/{body['session_id']}/checkpoint")
        assert ckpt.status_code == 200
        ckpt_id = ckpt.json()["checkpoint_id"]
        r = client.post(
            f"/v1/sessions/{body['session_id']}/restore",
            json={"checkpoint_id": ckpt_id},
        )
        assert r.status_code == 200

    def test_restore_unknown_checkpoint_404(self, client):
        body = self._make_session(client)
        r = client.post(
            f"/v1/sessions/{body['session_id']}/restore",
            json={"checkpoint_id": "ck_nope"},
        )
        assert r.status_code == 404

    def test_checkpoint_unknown_session_404(self, client):
        r = client.post("/v1/sessions/missing/checkpoint")
        assert r.status_code == 404

    def test_restore_unknown_session_404(self, client):
        r = client.post("/v1/sessions/missing/restore", json={"checkpoint_id": "ck"})
        assert r.status_code == 404

    def test_branch_creates_new_session(self, client):
        body = self._make_session(client)
        r = client.post(f"/v1/sessions/{body['session_id']}/branch")
        assert r.status_code == 201
        new = r.json()
        assert new["session_id"] != body["session_id"]
        assert new["parent_session_id"] == body["session_id"]

    def test_branch_unknown_session_404(self, client):
        r = client.post("/v1/sessions/missing/branch")
        assert r.status_code == 404

    def test_list_sessions_after_create(self, client):
        body = self._make_session(client)
        r = client.get("/v1/sessions")
        sids = [s["session_id"] for s in r.json()["sessions"]]
        assert body["session_id"] in sids


class TestRouteErrors:
    def test_world_init_error_returns_500(self, monkeypatch, client):
        from worldkernels.core.errors import WorldInitError

        def fake_load(*a, **kw):
            raise WorldInitError("foo", "boom")

        monkeypatch.setattr(client.app.state.engine, "load_model", fake_load)
        r = client.post("/v1/worlds", json={"model_id": "foo"})
        assert r.status_code == 500

    def test_vram_exhausted_returns_507(self, monkeypatch, client):
        from worldkernels.core.errors import VRAMExhaustedError

        def fake_create(*a, **kw):
            raise VRAMExhaustedError(1024.0, 256.0)

        monkeypatch.setattr(client.app.state.engine, "create_session", fake_create)
        r = client.post(
            "/v1/sessions",
            json={"world": "dummy", "height": 64, "width": 64, "frames_per_step": 1},
        )
        assert r.status_code == 507

    def test_step_session_terminated_returns_410(self, client):
        body = client.post(
            "/v1/sessions",
            json={"world": "dummy", "height": 32, "width": 32, "frames_per_step": 1},
        ).json()
        sess = client.app.state.engine.get_session(body["session_id"])
        sess.close()
        r = client.post(f"/v1/sessions/{body['session_id']}/step", json={"action_type": "null"})
        assert r.status_code == 410

    def test_step_world_kernel_error_returns_500(self, monkeypatch, client):
        from worldkernels.core.errors import WorldKernelError

        body = client.post(
            "/v1/sessions",
            json={"world": "dummy", "height": 32, "width": 32, "frames_per_step": 1},
        ).json()
        sess = client.app.state.engine.get_session(body["session_id"])

        def fake_transition(*a, **kw):
            raise WorldKernelError("internal")

        monkeypatch.setattr(sess._world, "transition", fake_transition)
        r = client.post(f"/v1/sessions/{body['session_id']}/step", json={"action_type": "null"})
        assert r.status_code == 500


class TestHelpers:
    def test_session_summary_shape(self, client):
        body_first = client.post(
            "/v1/sessions",
            json={"world": "dummy", "height": 32, "width": 32, "frames_per_step": 1},
        ).json()
        sess = client.app.state.engine.get_session(body_first["session_id"])
        summary = _session_summary(sess)
        for key in (  # noqa: E501
            "session_id",
            "world_id",
            "status",
            "step_index",
            "seed",
            "parent_session_id",
            "created_at",
        ):
            assert key in summary

    def test_observation_to_dict_with_frames(self):
        from worldkernels.core.observation import Observation
        from worldkernels.runtime.stages import StageTiming

        obs = Observation(
            step_index=1,
            generation_time_ms=5.0,
            frames=[b"\x01\x02"],
            stage_timing=StageTiming(encode_action_ms=1.0),
        )
        out = _observation_to_dict(obs, "s1")
        assert out["session_id"] == "s1"
        assert out["frames"] == ["AQI="]
        assert out["num_frames"] == 1
        assert out["stage_timing"]["encode_action_ms"] == 1.0

    def test_observation_to_dict_no_frames(self):
        from worldkernels.core.observation import Observation

        obs = Observation(step_index=0, generation_time_ms=0.0)
        out = _observation_to_dict(obs, "s1")
        assert "frames" not in out
        assert "num_frames" not in out
        assert "stage_timing" not in out

    def test_observation_to_dict_non_bytes_frame(self):
        from worldkernels.core.observation import Observation

        obs = Observation(step_index=0, generation_time_ms=0.0, frames=[42, b"x"])
        out = _observation_to_dict(obs, "s1")
        assert out["frames"][0] is None
        assert out["frames"][1] == "eA=="
