r"""Tests for the scheduler: batching, admission, preemption, dispatch."""

from __future__ import annotations

import pytest

from worldkernels.config import SchedulerConfig, WorldConfig
from worldkernels.core.request import StepRequest
from worldkernels.runtime.stages import TransitionMode
from worldkernels.scheduler import (
    AdmissionController,
    PreemptionCandidate,
    PreemptionPolicy,
    Scheduler,
    group_requests,
)


class _FakeWorld:
    transition_mode = TransitionMode.BIDIRECTIONAL

    def __init__(self, vram_mb: float = 100.0) -> None:
        self._vram = vram_mb

    def profile_vram(self, config: WorldConfig) -> float:
        return self._vram


class _FakeWorker:
    def __init__(self) -> None:
        self.batches: list[int] = []

    def execute(self, requests):
        self.batches.append(len(requests))
        return [(r.state, f"obs-{r.session_id}") for r in requests]


def _request(session_id: str, world: _FakeWorld) -> StepRequest:
    return StepRequest(session_id=session_id, world=world, state=None, action=None)


class TestBatching:
    def test_same_world_groups_together(self):
        world = _FakeWorld()
        reqs = [_request(f"s{i}", world) for i in range(3)]
        groups = group_requests(reqs, max_batch_size=8)
        assert len(groups) == 1
        assert groups[0].size == 3

    def test_distinct_worlds_separate_groups(self):
        reqs = [_request("a", _FakeWorld()), _request("b", _FakeWorld())]
        groups = group_requests(reqs, max_batch_size=8)
        assert len(groups) == 2

    def test_max_batch_size_splits_group(self):
        world = _FakeWorld()
        reqs = [_request(f"s{i}", world) for i in range(5)]
        groups = group_requests(reqs, max_batch_size=2)
        assert sorted(g.size for g in groups) == [1, 2, 2]


class TestAdmission:
    def _controller(self, **kw) -> AdmissionController:
        return AdmissionController(SchedulerConfig(**kw))

    def test_admits_when_vram_fits(self):
        decision = self._controller().check(
            _FakeWorld(vram_mb=500), WorldConfig(), live_sessions=0, free_vram_mb=4000
        )
        assert decision.admitted

    def test_refuses_when_vram_short(self):
        decision = self._controller(admission_headroom_mb=512).check(
            _FakeWorld(vram_mb=4000), WorldConfig(), live_sessions=0, free_vram_mb=4200
        )
        assert not decision.admitted
        assert "VRAM" in decision.reason

    def test_refuses_at_concurrency_cap(self):
        decision = self._controller(max_concurrent_sessions=2).check(
            _FakeWorld(), WorldConfig(), live_sessions=2, free_vram_mb=None
        )
        assert not decision.admitted
        assert "concurrency" in decision.reason

    def test_cpu_skips_vram_check(self):
        decision = self._controller().check(
            _FakeWorld(vram_mb=1e9), WorldConfig(), live_sessions=0, free_vram_mb=None
        )
        assert decision.admitted


class TestPreemption:
    def test_selects_lowest_priority_then_oldest(self):
        policy = PreemptionPolicy(mode="swap")
        candidates = [
            PreemptionCandidate("a", last_active_step=10, priority=1),
            PreemptionCandidate("b", last_active_step=2, priority=0),
            PreemptionCandidate("c", last_active_step=8, priority=0),
        ]
        decision = policy.select_victim(candidates)
        assert decision is not None
        assert decision.victim_id == "b"
        assert decision.mode == "swap"

    def test_no_candidates_returns_none(self):
        assert PreemptionPolicy().select_victim([]) is None


class TestSchedulerDispatch:
    def test_step_passthrough(self):
        sched = Scheduler(_FakeWorker())
        state, obs = sched.step(
            world=_FakeWorld(), state="st", action=None, modalities=["frames"], step_index=0
        )
        assert obs == "obs-"

    def test_run_scheduled_batches_compatible_requests(self):
        worker = _FakeWorker()
        sched = Scheduler(worker, SchedulerConfig(max_batch_size=8))
        world = _FakeWorld()
        for i in range(3):
            sched.add_request(_request(f"s{i}", world))
        assert sched.pending == 3
        results = sched.run_scheduled()
        assert set(results) == {"s0", "s1", "s2"}
        assert worker.batches == [3]
        assert sched.pending == 0

    def test_run_scheduled_separates_incompatible(self):
        worker = _FakeWorker()
        sched = Scheduler(worker)
        sched.add_request(_request("a", _FakeWorld()))
        sched.add_request(_request("b", _FakeWorld()))
        sched.run_scheduled()
        assert sorted(worker.batches) == [1, 1]


def test_group_requests_rejects_bad_batch_size():
    with pytest.raises(ValueError, match="max_batch_size"):
        group_requests([], max_batch_size=0)
