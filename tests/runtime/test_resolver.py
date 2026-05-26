r"""Resolver: SharedPlan vs IsolatedPlan decision (ADR-012)."""

from __future__ import annotations

import shutil

import pytest

from worldkernels.runtime import resolver
from worldkernels.worlds.hub import ModelCard


def _card(adapter: str, constraints: list[str] | None = None, isolation: str = "auto") -> ModelCard:
    return ModelCard(
        adapter=adapter,
        constraints=constraints or [],
        isolation=isolation,  # type: ignore[arg-type]
    )


class TestResolveInstallPlan:
    def test_no_constraints_returns_shared(self):
        card = _card("dummy")
        plan = resolver.resolve_install_plan(card, [])
        assert isinstance(plan, resolver.SharedPlan)

    def test_forced_isolated_short_circuits(self):
        card = _card("dummy", isolation="isolated", constraints=["torch>=2.0"])
        plan = resolver.resolve_install_plan(card, [])
        assert isinstance(plan, resolver.IsolatedPlan)
        assert "isolated" in plan.reason

    @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
    def test_overlapping_ranges_share_env(self):
        a = _card("a", constraints=["packaging>=20.0"])
        b = _card("b", constraints=["packaging>=21.0"])
        plan = resolver.resolve_install_plan(b, [a])
        assert isinstance(plan, resolver.SharedPlan)

    @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
    def test_conflicting_pins_force_isolation(self):
        a = _card("a", constraints=["packaging==20.0"])
        b = _card("b", constraints=["packaging==99.0"])
        plan = resolver.resolve_install_plan(b, [a])
        assert isinstance(plan, resolver.IsolatedPlan)

    @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
    def test_shared_forced_with_conflict_raises(self):
        a = _card("a", constraints=["packaging==20.0"])
        b = _card("b", constraints=["packaging==99.0"], isolation="shared")
        with pytest.raises(resolver.IncompatibleDependenciesError):
            resolver.resolve_install_plan(b, [a])

    def test_uv_missing_falls_back_to_shared(self, monkeypatch):
        monkeypatch.setattr(resolver.shutil, "which", lambda _: None)
        a = _card("a", constraints=["packaging>=20.0"])
        b = _card("b", constraints=["packaging>=21.0"])
        plan = resolver.resolve_install_plan(b, [a])
        assert isinstance(plan, resolver.SharedPlan)
