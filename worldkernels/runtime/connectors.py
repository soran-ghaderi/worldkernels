r"""Inter-stage data transport for the world model pipeline."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from worldkernels.runtime.stages import StageOutput

log = logging.getLogger(__name__)


class StageConnector(ABC):
    r"""Abstract transport between pipeline stages."""

    @abstractmethod
    def put(
        self,
        from_stage: str,
        to_stage: str,
        key: str,
        data: StageOutput,
    ) -> bool:
        r"""Store a stage output for the destination stage."""

    @abstractmethod
    def get(
        self,
        from_stage: str,
        to_stage: str,
        key: str,
    ) -> StageOutput | None:
        r"""Retrieve a stage output. Returns None if not yet available."""

    @abstractmethod
    def cleanup(self, key: str) -> None:
        r"""Release all resources associated with a key."""


class LocalConnector(StageConnector):
    r"""Zero-copy connector for co-located stages on the same device."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str, str], StageOutput] = {}

    def put(
        self,
        from_stage: str,
        to_stage: str,
        key: str,
        data: StageOutput,
    ) -> bool:
        self._store[(from_stage, to_stage, key)] = data
        return True

    def get(
        self,
        from_stage: str,
        to_stage: str,
        key: str,
    ) -> StageOutput | None:
        return self._store.pop((from_stage, to_stage, key), None)

    def cleanup(self, key: str) -> None:
        to_remove = [k for k in self._store if k[2] == key]
        for k in to_remove:
            del self._store[k]


class ConnectorRegistry:
    r"""Factory for creating stage connector instances."""

    _registry: dict[str, type[StageConnector]] = {}

    @classmethod
    def register(cls, name: str, connector_cls: type[StageConnector]) -> None:
        if name in cls._registry:
            log.warning("Overwriting connector registration: %s", name)
        cls._registry[name] = connector_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> StageConnector:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown connector '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_registered(cls) -> list[str]:
        return list(cls._registry.keys())


# Built-in registrations
ConnectorRegistry.register("local", LocalConnector)
