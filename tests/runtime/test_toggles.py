r"""Per-toggle behavior assertions: every flag controls observable behavior.

For each flag, we load an engine with it ON vs OFF, run the relevant code
path, and assert the subsystem was either instantiated and exercised, or
absent. Output shapes/types stay identical (graceful fallback) — only the
side-channel evidence differs.
"""

from __future__ import annotations

import torch

from worldkernels import Action, WorldConfig, WorldEngine
from worldkernels.config import RuntimeConfig
from worldkernels.runtime.memory.kv_cache import KVCacheManager
from worldkernels.runtime.memory.latent_pool import LatentPool
from worldkernels.runtime.memory.trajectory_cache import TrajectoryCache


def _step_dummy(engine: WorldEngine) -> None:
    engine.load_model("dummy")
    session = engine.create_session("dummy", config=WorldConfig(height=32, width=32))
    session.step(Action("null", {}), modalities=["frames"])
    session.close()


class TestLatentPool:
    def test_on_creates_pool_and_records_use(self):
        wk = WorldEngine(RuntimeConfig(latent_pool=True), device="cpu")
        try:
            assert isinstance(wk._worker.runner.pool, LatentPool)
            _step_dummy(wk)
            # at least one acquire/release pair from DummyWorld.transition
            assert wk._worker.runner.pool.num_pooled >= 1
        finally:
            wk.shutdown()

    def test_off_pool_is_none(self):
        wk = WorldEngine(RuntimeConfig(latent_pool=False), device="cpu")
        try:
            assert wk._worker.runner.pool is None
            _step_dummy(wk)  # still works via the fallback path
        finally:
            wk.shutdown()


class TestTeaCache:
    def test_on_creates_cache_and_records_activity(self):
        wk = WorldEngine(RuntimeConfig(teacache=True), device="cpu")
        try:
            cache = wk._worker.runner.teacache
            assert cache is not None
            _step_dummy(wk)
            assert (cache.hits + cache.misses) >= 1
        finally:
            wk.shutdown()

    def test_off_no_cache(self):
        wk = WorldEngine(RuntimeConfig(teacache=False), device="cpu")
        try:
            assert wk._worker.runner.teacache is None
            _step_dummy(wk)  # fallback path unchanged
        finally:
            wk.shutdown()


class TestTrajectoryCache:
    def test_on_creates_cache_and_match_returns_match(self):
        wk = WorldEngine(RuntimeConfig(trajectory_cache=True), device="cpu")
        try:
            assert isinstance(wk.trajectory_cache, TrajectoryCache)
            assert wk.cache_trajectory(["a", "b", "c"], [10, 11, 12])
            match = wk.lookup_trajectory_prefix(["a", "b", "d"])
            assert match is not None
            assert match.length == 2
            assert match.block_ids == [10, 11]
        finally:
            wk.shutdown()

    def test_off_lookup_returns_none(self):
        wk = WorldEngine(RuntimeConfig(trajectory_cache=False), device="cpu")
        try:
            assert wk.trajectory_cache is None
            assert wk.lookup_trajectory_prefix(["a"]) is None
            assert wk.cache_trajectory(["a"], [1]) is False
        finally:
            wk.shutdown()


class TestKvCachePaged:
    def test_on_ensures_kv_manager_for_causal_world(self, monkeypatch):
        from tests._helpers.mocks import MockWorld
        from worldkernels.worlds import registry as reg

        class CausalWorld(MockWorld):
            supports_kv_cache = True

        reg.register_world("_causal_world", CausalWorld)
        try:
            wk = WorldEngine(RuntimeConfig(kv_cache_paged=True), device="cpu")
            try:
                wk.load_model("_causal_world")
                assert isinstance(wk._worker.runner.kv_cache, KVCacheManager)
                assert wk._worker.runner.block_manager is not None
            finally:
                wk.shutdown()
        finally:
            reg._REGISTRY.pop("_causal_world", None)

    def test_off_kv_manager_stays_none(self, monkeypatch):
        from tests._helpers.mocks import MockWorld
        from worldkernels.worlds import registry as reg

        class CausalWorld(MockWorld):
            supports_kv_cache = True

        reg.register_world("_causal_world_off", CausalWorld)
        try:
            wk = WorldEngine(RuntimeConfig(kv_cache_paged=False), device="cpu")
            try:
                wk.load_model("_causal_world_off")
                assert wk._worker.runner.kv_cache is None
                assert wk._worker.runner.block_manager is None
            finally:
                wk.shutdown()
        finally:
            reg._REGISTRY.pop("_causal_world_off", None)


class TestIterationBatching:
    def test_on_drives_transition_iter(self, monkeypatch):
        wk = WorldEngine(RuntimeConfig(iteration_batching=True), device="cpu")
        try:
            wk.load_model("dummy")
            world = wk._worlds["dummy"]
            calls = {"iter": 0, "atomic": 0}
            orig_iter = world.transition_iter
            orig_t = world.transition

            def spy_iter(state, action):
                calls["iter"] += 1
                yield from orig_iter(state, action)

            def spy_t(state, action):
                calls["atomic"] += 1
                return orig_t(state, action)

            monkeypatch.setattr(world, "transition_iter", spy_iter)
            monkeypatch.setattr(world, "transition", spy_t)
            session = wk.create_session("dummy", config=WorldConfig(height=32, width=32))
            session.step(Action("null", {}), modalities=["frames"])
            assert calls["iter"] == 1
            assert calls["atomic"] == 0
        finally:
            wk.shutdown()

    def test_off_drives_atomic_transition(self, monkeypatch):
        wk = WorldEngine(RuntimeConfig(iteration_batching=False), device="cpu")
        try:
            wk.load_model("dummy")
            world = wk._worlds["dummy"]
            calls = {"iter": 0, "atomic": 0}
            orig_iter = world.transition_iter
            orig_t = world.transition

            def spy_iter(state, action):
                calls["iter"] += 1
                yield from orig_iter(state, action)

            def spy_t(state, action):
                calls["atomic"] += 1
                return orig_t(state, action)

            monkeypatch.setattr(world, "transition_iter", spy_iter)
            monkeypatch.setattr(world, "transition", spy_t)
            session = wk.create_session("dummy", config=WorldConfig(height=32, width=32))
            session.step(Action("null", {}), modalities=["frames"])
            assert calls["iter"] == 0
            assert calls["atomic"] == 1
        finally:
            wk.shutdown()


class TestOffloadIdle:
    def test_on_offloads_non_active(self):
        import asyncio

        async def scenario():
            from worldkernels.core.session import LatentState
            from worldkernels.engine import AsyncEngine

            wk = WorldEngine(RuntimeConfig(offload_idle=True), device="cpu")
            try:
                wk.load_model("dummy")
                ae = AsyncEngine(wk, batch_window=0.001)
                cfg = WorldConfig(height=32, width=32)
                sess_a = wk.create_session("dummy", config=cfg)
                sess_b = wk.create_session("dummy", config=cfg)
                sess_b.state = LatentState(data=sess_b.state.data, device="cuda:0")
                await ae.step(sess_a.session_id, Action("null", {}))
                await asyncio.sleep(0.01)
                assert ae.offload_count >= 1
                await ae.shutdown()
            finally:
                wk.shutdown()

        asyncio.run(scenario())

    def test_off_does_not_offload(self):
        import asyncio

        async def scenario():
            from worldkernels.core.session import LatentState
            from worldkernels.engine import AsyncEngine

            wk = WorldEngine(RuntimeConfig(offload_idle=False), device="cpu")
            try:
                wk.load_model("dummy")
                ae = AsyncEngine(wk, batch_window=0.001)
                cfg = WorldConfig(height=32, width=32)
                sess_a = wk.create_session("dummy", config=cfg)
                sess_b = wk.create_session("dummy", config=cfg)
                sess_b.state = LatentState(data=sess_b.state.data, device="cuda:0")
                await ae.step(sess_a.session_id, Action("null", {}))
                await asyncio.sleep(0.01)
                assert ae.offload_count == 0
                await ae.shutdown()
            finally:
                wk.shutdown()

        asyncio.run(scenario())


class TestAttentionBackend:
    def test_selector_reads_forward_context_sdpa(self):
        from worldkernels.runtime.attention.backends import SDPABackend
        from worldkernels.runtime.attention.selector import select_attention_backend
        from worldkernels.runtime.forward_context import ForwardContext, set_forward_context

        with set_forward_context(ForwardContext(attention_backend="sdpa")):
            backend = select_attention_backend()
        assert isinstance(backend, SDPABackend)

    def test_selector_falls_back_to_platform_default_when_ctx_none(self):
        from worldkernels.runtime.attention.selector import select_attention_backend
        from worldkernels.runtime.forward_context import ForwardContext, set_forward_context

        with set_forward_context(ForwardContext(attention_backend=None)):
            backend = select_attention_backend()
        assert backend is not None  # one of SDPA / Flash

    def test_runner_threads_attention_backend_into_context(self):
        wk = WorldEngine(
            RuntimeConfig(attention_backend="sdpa", iteration_batching=False), device="cpu"
        )
        try:
            wk.load_model("dummy")
            seen = {}

            from worldkernels.runtime.forward_context import get_forward_context

            orig = wk._worlds["dummy"].transition

            def spy(state, action):
                ctx = get_forward_context()
                seen["backend"] = ctx.attention_backend
                return orig(state, action)

            wk._worlds["dummy"].transition = spy
            session = wk.create_session("dummy", config=WorldConfig(height=32, width=32))
            session.step(Action("null", {}), modalities=["frames"])
            assert seen["backend"] == "sdpa"
        finally:
            wk.shutdown()


class TestQuantization:
    def test_none_does_not_call_registry(self, monkeypatch):
        from worldkernels.runtime.quantization.registry import QuantizationRegistry

        called = {"n": 0}

        def spy(self, module, scheme):
            called["n"] += 1
            return module

        monkeypatch.setattr(QuantizationRegistry, "apply", spy)
        wk = WorldEngine(RuntimeConfig(quantization="none"), device="cpu")
        try:
            wk.load_model("dummy")
            assert called["n"] == 0
        finally:
            wk.shutdown()

    def test_int8_invokes_registry_with_target(self, monkeypatch):
        from worldkernels.runtime.quantization.registry import QuantizationRegistry

        recorded = {}

        def spy(self, module, scheme):
            recorded["scheme"] = scheme
            recorded["module"] = module
            return module

        monkeypatch.setattr(QuantizationRegistry, "apply", spy)
        wk = WorldEngine(RuntimeConfig(quantization="int8"), device="cpu")
        try:
            wk.load_model("dummy")
            assert recorded["scheme"] == "int8"
            assert isinstance(recorded["module"], torch.nn.Linear)
        finally:
            wk.shutdown()


class TestSessionOverrides:
    def test_session_teacache_override_disables_recording(self):
        wk = WorldEngine(RuntimeConfig(teacache=True), device="cpu")
        try:
            wk.load_model("dummy")
            cfg = WorldConfig(height=32, width=32)
            s_off = wk.create_session("dummy", config=cfg, overrides={"teacache": False})
            s_on = wk.create_session("dummy", config=cfg)
            # cache exists at the runner level
            cache = wk._worker.runner.teacache
            before = cache.hits + cache.misses
            s_off.step(Action("null", {}), modalities=["frames"])
            mid = cache.hits + cache.misses
            s_on.step(Action("null", {}), modalities=["frames"])
            after = cache.hits + cache.misses
            assert mid == before     # session-off didn't touch the cache
            assert after > mid       # session-on did
        finally:
            wk.shutdown()

    def test_session_iteration_batching_override(self, monkeypatch):
        wk = WorldEngine(RuntimeConfig(iteration_batching=True), device="cpu")
        try:
            wk.load_model("dummy")
            world = wk._worlds["dummy"]
            calls = {"iter": 0, "atomic": 0}
            orig_iter = world.transition_iter
            orig_t = world.transition

            def spy_iter(state, action):
                calls["iter"] += 1
                yield from orig_iter(state, action)

            def spy_t(state, action):
                calls["atomic"] += 1
                return orig_t(state, action)

            monkeypatch.setattr(world, "transition_iter", spy_iter)
            monkeypatch.setattr(world, "transition", spy_t)

            cfg = WorldConfig(height=32, width=32)
            s = wk.create_session("dummy", config=cfg, overrides={"iteration_batching": False})
            s.step(Action("null", {}), modalities=["frames"])
            assert calls["atomic"] == 1
            assert calls["iter"] == 0
        finally:
            wk.shutdown()

    def test_unsafe_override_warns_and_drops(self, caplog):
        import logging

        wk = WorldEngine(device="cpu")
        try:
            wk.load_model("dummy")
            with caplog.at_level(logging.WARNING):
                s = wk.create_session("dummy", overrides={"torch_compile": False})
            assert s.overrides is None or "torch_compile" not in s.overrides
            assert any("torch_compile" in r.message for r in caplog.records)
        finally:
            wk.shutdown()


class TestGracefulFallbackShapes:
    def test_dummy_step_output_shape_invariant_under_pool(self):
        outputs = []
        for pool_on in (True, False):
            wk = WorldEngine(RuntimeConfig(latent_pool=pool_on), device="cpu")
            try:
                wk.load_model("dummy")
                cfg = WorldConfig(height=32, width=32)
                session = wk.create_session("dummy", config=cfg)
                obs = session.step(Action("null", {}), modalities=["frames"])
                outputs.append((len(obs.frames) if obs.frames else 0, obs.step_index))
                session.close()
            finally:
                wk.shutdown()
        assert outputs[0] == outputs[1]
