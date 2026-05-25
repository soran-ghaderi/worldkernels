r"""Scheduler layer: continuous-batching dispatch, admission, preemption."""

from __future__ import annotations

from worldkernels.scheduler.admission import AdmissionController, AdmissionDecision
from worldkernels.scheduler.batching import (
    CompatibilityGroup,
    CompatibilityKey,
    group_requests,
)
from worldkernels.scheduler.preemption import (
    PreemptionCandidate,
    PreemptionDecision,
    PreemptionPolicy,
)
from worldkernels.scheduler.scheduler import Scheduler

__all__ = [
    "Scheduler",
    "CompatibilityKey",
    "CompatibilityGroup",
    "group_requests",
    "AdmissionController",
    "AdmissionDecision",
    "PreemptionPolicy",
    "PreemptionCandidate",
    "PreemptionDecision",
]
