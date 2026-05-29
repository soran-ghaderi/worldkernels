r"""WorldEngine — the request-and-lifecycle layer.

Owns the world registry and session registry, resolves and loads models, and
creates sessions. Execution is delegated to a `Scheduler`, which
dispatches to a `Worker`; the engine itself runs no model compute.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from worldkernels.core.errors import (
    SessionLimitError,
    VRAMExhaustedError,
    WorldAlreadyLoadedError,
    WorldInitError,
    WorldNotFoundError,
)

if TYPE_CHECKING:
    from worldkernels.config import RuntimeConfig, WorldConfig
    from worldkernels.core.session import Session
    from worldkernels.worlds.base import WorldModel

log = logging.getLogger(__name__)


def _default_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability() >= (8, 0):
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return _default_dtype(device)
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


class WorldEngine:
    r"""GPU-first world-model simulation engine.

    Args:
        device: Target device (``"cuda"``, ``"cpu"``, or ``"cuda:N"``).
        max_sessions: Maximum concurrent sessions.
        offload_idle: Whether to offload idle session state.
    """

    def __init__(
        self,
        config: "RuntimeConfig | str | None" = None,
        *,
        device: str | None = None,
        max_sessions: int | None = None,
        offload_idle: bool | None = None,
    ) -> None:
        r"""Construct the engine from a `RuntimeConfig` (or profile name).

        Args:
            config: A `RuntimeConfig`, a profile name (``"baseline"`` etc.), or
                ``None`` for resolved defaults (honoring ``WK_*`` env vars).
            device: Back-compat override of ``config.device``.
            max_sessions: Back-compat override of ``config.max_sessions``.
            offload_idle: Back-compat override of ``config.offload_idle``.
        """
        from worldkernels.config.active import set_active_config
        from worldkernels.config.profiles import resolve_runtime_config
        from worldkernels.config.runtime import RuntimeConfig

        if isinstance(config, RuntimeConfig):
            cfg = config
        elif isinstance(config, str):
            cfg, _ = resolve_runtime_config(profile=config)
        else:
            cfg, _ = resolve_runtime_config()

        if device is not None:
            cfg.device = device
        if max_sessions is not None:
            cfg.max_sessions = max_sessions
        if offload_idle is not None:
            cfg.offload_idle = offload_idle

        self.config = cfg
        set_active_config(cfg)

        self.device = cfg.device
        self.max_sessions = cfg.max_sessions
        self.offload_idle = cfg.offload_idle
        self.dtype = _resolve_dtype(cfg.dtype, cfg.device)

        self._worlds: dict[str, WorldModel] = {}
        self._sessions: dict[str, Session] = {}
        self._tiers: dict[str, str] = {}
        self._cards: dict[str, Any] = {}

        from worldkernels.scheduler import Scheduler
        from worldkernels.worker import Worker

        self._worker = Worker(device=self.device, dtype=self.dtype, parallel_config=cfg.parallel)
        self._scheduler = Scheduler(self._worker, config=cfg.scheduler)

        log.info(
            "WorldEngine initialized: device=%s, dtype=%s, max_sessions=%d",
            self.device,
            self.dtype,
            self.max_sessions,
        )

    def load_model(
        self,
        model_id: str,
        alias: str | None = None,
        variant: str | None = None,
        ckpt_path: str | None = None,
        progress: Any = None,
        allow_fetch: bool = True,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""Load a world model from a hub alias, HF repo id, or local checkpoint path.

        Drives the resolver (ADR-012): the resolver decides whether the model can
        share the current env or needs an isolated subprocess. Shared models go
        through `worldkernels.bootstrap.prepare` and instantiate locally; isolated
        models materialize a per-model uv venv and are accessed via `RemoteWorld`.

        Args:
            model_id: Short alias, HF repo id, HF URL, or local path.
            alias: Override the in-engine key (defaults to a slug of ``model_id``).
            variant: Pick a variant from the model card.
            ckpt_path: Bypass HF download with a local checkpoint file.
            progress: Optional `ProgressController` (CLI/HTTP pass theirs through).
            allow_fetch: If ``False``, errors instead of installing/cloning/downloading.
            trust_remote_code: Allow custom code from HF Hub (reserved).
            **kwargs: Forwarded to the world constructor (overrides card defaults).
        """
        from worldkernels.bootstrap.errors import ModelNotFoundError as _ModelNotFoundError
        from worldkernels.bootstrap.resolve import resolve as _resolve_ref
        from worldkernels.config import WorldConfig as WC
        from worldkernels.runtime.resolver import IsolatedPlan, resolve_install_plan

        try:
            resolved = _resolve_ref(model_id, variant=variant, ckpt_path=ckpt_path)
        except _ModelNotFoundError as exc:
            raise WorldNotFoundError(model_id) from exc
        card = resolved.card
        if card.isolation == "auto" and self.config.isolation != "auto":
            import dataclasses

            card = dataclasses.replace(card, isolation=self.config.isolation)
            resolved = dataclasses.replace(resolved, card=card)
        plan = resolve_install_plan(card, list(self._cards.values()))

        if isinstance(plan, IsolatedPlan):
            world, key, tier = self._load_isolated(
                model_id, resolved, plan, alias, progress, allow_fetch, kwargs
            )
        else:
            world, key, tier = self._load_shared(
                model_id, alias, variant, ckpt_path, progress, allow_fetch, kwargs
            )

        if key in self._worlds:
            try:
                world.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            raise WorldAlreadyLoadedError(key)

        try:
            world.warmup(getattr(world, "default_config", None) or WC())
        except Exception as exc:
            raise WorldInitError(model_id, f"warmup failed: {exc}") from exc

        self._worlds[key] = world
        self._tiers[key] = tier
        self._cards[key] = card
        _emit_tier_metric(key, tier)
        _emit_worker_count(self._tiers)
        log.info("Loaded world: %s (tier=%s, class=%s)", key, tier, type(world).__name__)

    def _load_shared(
        self,
        model_id: str,
        alias: str | None,
        variant: str | None,
        ckpt_path: str | None,
        progress: Any,
        allow_fetch: bool,
        kwargs: dict[str, Any],
    ) -> tuple[Any, str, str]:
        from worldkernels.bootstrap import prepare
        from worldkernels.worlds.registry import get_world_class

        prepared = prepare(
            model_id,
            variant=variant,
            ckpt_path=ckpt_path,
            progress=progress,
            allow_fetch=allow_fetch,
            **kwargs,
        )
        key = alias or prepared.alias
        try:
            world_cls = get_world_class(prepared.adapter)
        except KeyError as exc:
            raise WorldNotFoundError(model_id) from exc

        world = world_cls(**prepared.kwargs)
        try:
            world.initialize(device=self.device, dtype=self.dtype)
        except Exception as exc:
            raise WorldInitError(model_id, str(exc)) from exc
        return world, key, "shared"

    def _load_isolated(
        self,
        model_id: str,
        resolved: Any,
        plan: Any,
        alias: str | None,
        progress: Any,
        allow_fetch: bool,
        kwargs: dict[str, Any],
    ) -> tuple[Any, str, str]:
        from worldkernels.runtime import envs
        from worldkernels.worlds.remote import RemoteWorld

        if progress is not None:
            progress.event("isolating", "running", f"reason: {plan.reason}")

        requirements = list(plan.requirements)
        if "worldkernels" not in " ".join(requirements):
            requirements.append("worldkernels")

        envs.materialize_env(
            model_id,
            requirements,
            device=self.device,
            progress=progress,
            allow_fetch=allow_fetch,
        )

        ctor_kwargs: dict[str, Any] = dict(resolved.card.default_kwargs)
        if resolved.variant is not None:
            ctor_kwargs["variant"] = resolved.variant
        if resolved.ckpt_path is not None:
            ctor_kwargs["ckpt_path"] = resolved.ckpt_path
        ctor_kwargs.update(kwargs)

        key = alias or self._alias_for(model_id, resolved.variant)
        world = RemoteWorld(
            model_id=model_id,
            adapter=resolved.card.adapter,
            ctor_kwargs=ctor_kwargs,
        )
        try:
            world.initialize(device=self.device, dtype=self.dtype)
        except Exception as exc:
            try:
                world.close()
            finally:
                raise WorldInitError(model_id, str(exc)) from exc
        return world, key, "isolated"

    @staticmethod
    def _alias_for(model_id: str, variant: str | None) -> str:
        base = model_id.split("/")[-1]
        return f"{base}:{variant}" if variant else base

    def create_session(
        self,
        world: str,
        config: WorldConfig | None = None,
        seed: int | None = None,
    ) -> Session:
        r"""Create a new simulation session bound to a loaded world model."""
        from worldkernels.config import WorldConfig as WC
        from worldkernels.core.session import Session

        if world not in self._worlds:
            raise WorldNotFoundError(world)
        if len(self._sessions) >= self.max_sessions:
            raise SessionLimitError(self.max_sessions)

        cfg = config or WC()
        actual_seed = seed if seed is not None else 0
        world_instance = self._worlds[world]

        vram_mb = world_instance.profile_vram(cfg)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
            if vram_mb > free_mb:
                raise VRAMExhaustedError(required_mb=vram_mb, available_mb=free_mb)

        initial_state = world_instance.create_initial_state(cfg, actual_seed)
        session = Session(
            world_id=world,
            config=cfg,
            state=initial_state,
            seed=actual_seed,
            _world=world_instance,
            _scheduler=self._scheduler,
        )
        self._sessions[session.session_id] = session
        log.info("Created session: %s (world=%s)", session.session_id, world)
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def list_worlds(self) -> list[str]:
        return list(self._worlds.keys())

    def unload_model(self, name: str) -> None:
        r"""Unload a world model and close all its sessions."""
        if name not in self._worlds:
            raise WorldNotFoundError(name)
        for sid in [s for s, sess in self._sessions.items() if sess.world_id == name]:
            self.close_session(sid)
        world = self._worlds.pop(name)
        self._tiers.pop(name, None)
        self._cards.pop(name, None)
        close = getattr(world, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                log.warning("error closing world %r: %s", name, exc)
        _emit_worker_count(self._tiers)
        log.info("Unloaded world: %s", name)

    def get_tier(self, name: str) -> str | None:
        r"""Return the isolation tier (``"shared"`` / ``"isolated"``) for a loaded model."""
        return self._tiers.get(name)

    def list_tiers(self) -> dict[str, str]:
        return dict(self._tiers)

    def shutdown(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        for name, world in list(self._worlds.items()):
            close = getattr(world, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    log.warning("error closing world %r during shutdown: %s", name, exc)
        self._worlds.clear()
        self._tiers.clear()
        self._cards.clear()
        log.info("WorldEngine shut down.")


def _emit_tier_metric(model_id: str, tier: str) -> None:
    try:
        from worldkernels.runtime import metrics

        metrics.set_isolation_tier(model_id, 0 if tier == "shared" else 1)
    except Exception:
        pass


def _emit_worker_count(tiers: dict[str, str]) -> None:
    try:
        from worldkernels.runtime import metrics

        metrics.set_worker_processes(sum(1 for t in tiers.values() if t == "isolated"))
    except Exception:
        pass
