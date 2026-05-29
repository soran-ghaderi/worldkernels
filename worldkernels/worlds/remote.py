r"""Out-of-process world proxy (ADR-012 Tier 2).

``RemoteWorld`` implements the same `InteractiveWorldModel` interface as an
in-process world but forwards every method to a subprocess via Unix-socket RPC
and CUDA IPC (with host-shm fallback). The engine + scheduler cannot tell the
difference; the executor doesn't distinguish.

The worker process lives in a per-model isolated venv (see ``runtime/envs.py``).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from worldkernels.core.errors import WorldKernelError
from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState
from worldkernels.runtime import envs, ipc
from worldkernels.runtime.rpc import RpcClient, default_socket_path
from worldkernels.worlds.base import InteractiveWorldModel

if TYPE_CHECKING:
    from worldkernels.config import WorldConfig
    from worldkernels.core.action import Action

log = logging.getLogger(__name__)

__all__ = ["RemoteWorld", "WorkerHandle"]


@dataclass
class WorkerHandle:
    process: subprocess.Popen
    socket_path: Path
    client: RpcClient


class RemoteWorld(InteractiveWorldModel):
    r"""Proxy to a `WorldModel` running in an isolated subprocess."""

    name = "remote"

    def __init__(
        self,
        model_id: str,
        adapter: str,
        ctor_kwargs: dict[str, Any] | None = None,
        env_root: Path | None = None,
        worker_module: str = "worldkernels.runtime.worker",
        log_level: str = "INFO",
    ) -> None:
        self.model_id = model_id
        self.adapter = adapter
        self.ctor_kwargs = ctor_kwargs or {}
        self.env_root = env_root or envs.env_path(model_id)
        self.worker_module = worker_module
        self.log_level = log_level
        self.device: str = "cuda"
        self.dtype: torch.dtype = torch.bfloat16
        self._worker: WorkerHandle | None = None

    def _spawn(self) -> WorkerHandle:
        socket_path = default_socket_path(self.model_id)
        py = envs.venv_python(self.model_id)
        if not py.exists():
            py_str = "python"
        else:
            py_str = str(py)
        cmd = [
            py_str,
            "-m",
            self.worker_module,
            "--socket",
            str(socket_path),
            "--log-level",
            self.log_level,
        ]
        log.info("spawning worker: %s", " ".join(cmd))
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(cmd, env=env)
        client = RpcClient(socket_path)
        try:
            client.connect()
        except ConnectionError:
            proc.terminate()
            raise
        return WorkerHandle(process=proc, socket_path=socket_path, client=client)

    def _ensure_worker(self) -> WorkerHandle:
        if self._worker is None or self._worker.process.poll() is not None:
            if self._worker is not None:
                log.warning(
                    "worker for %s died (exit=%s); respawning",
                    self.model_id,
                    self._worker.process.returncode,
                )
                try:
                    from worldkernels.runtime import metrics

                    metrics.inc_worker_respawns(self.model_id)
                except Exception:
                    pass
            self._worker = self._spawn()
        return self._worker

    def initialize(self, device: str, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        handle = self._ensure_worker()
        handle.client.call(
            "init",
            adapter=self.adapter,
            kwargs=self.ctor_kwargs,
            device=device,
            dtype=str(dtype),
        )

    def warmup(self, config: "WorldConfig") -> None:
        handle = self._ensure_worker()
        handle.client.call("warmup", config=_config_to_dict(config))

    def encode_action(self, action: "Action") -> torch.Tensor:
        handle = self._ensure_worker()
        res = handle.client.call(
            "encode_action",
            action_type=action.action_type,
            payload=action.payload,
        )
        return _RemoteRef(self, res["handle"])  # type: ignore[return-value]

    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        handle = self._ensure_worker()
        state_h = _ref_handle(state.data)
        action_h = _ref_handle(action_encoded)
        res = handle.client.call("transition", state_handle=state_h, action_handle=action_h)
        return LatentState(data=_RemoteRef(self, res["handle"]), device=self.device)

    def decode_observation(self, state: LatentState, modalities: list[str]) -> Observation:
        handle = self._ensure_worker()
        state_h = _ref_handle(state.data)
        res = handle.client.call("decode", state_handle=state_h, modalities=modalities)
        frames: list[Any] | None = None
        raw = res.get("frames")
        if raw is not None:
            frames = [_unwrap_frame(f) for f in raw]
        latent = None
        if res.get("latent_ipc") is not None:
            latent = ipc.recv_tensor(_dict_to_ipc(res["latent_ipc"]))
        return Observation(
            step_index=int(res.get("step_index", 0)),
            generation_time_ms=float(res.get("generation_time_ms", 0.0)),
            frames=frames,
            latent=latent,
        )

    def create_initial_state(self, config: "WorldConfig", seed: int) -> LatentState:
        handle = self._ensure_worker()
        res = handle.client.call(
            "create_initial_state", config=_config_to_dict(config), seed=int(seed)
        )
        return LatentState(data=_RemoteRef(self, res["handle"]), device=self.device)

    def profile_vram(self, config: "WorldConfig") -> float:
        handle = self._ensure_worker()
        res = handle.client.call("profile_vram", config=_config_to_dict(config))
        return float(res.get("vram_mb", 0.0))

    def health(self) -> dict[str, Any]:
        handle = self._ensure_worker()
        return handle.client.call("health")

    def close(self) -> None:
        if self._worker is None:
            return
        try:
            self._worker.client.call("close")
        except Exception as exc:
            log.debug("worker close errored: %s", exc)
        finally:
            self._worker.client.close()
            self._terminate(self._worker.process)
            self._worker = None

    def _terminate(self, proc: subprocess.Popen, timeout: float = 5.0) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=2.0)


@dataclass
class _RemoteRef:
    r"""Opaque server-side handle. Looks like a tensor/state to the engine."""

    world: RemoteWorld
    handle: str

    def release(self) -> None:
        try:
            self.world._ensure_worker().client.call("release", handle=self.handle)
        except Exception:
            pass


def _ref_handle(obj: Any) -> str:
    if isinstance(obj, _RemoteRef):
        return obj.handle
    raise WorldKernelError(f"remote world expected a server-side handle, got {type(obj).__name__}")


def _unwrap_frame(envelope: dict[str, Any]) -> Any:
    kind = envelope.get("kind")
    if kind == "bytes":
        return bytes(envelope["data"])
    if kind == "ipc":
        return ipc.recv_tensor(_dict_to_ipc(envelope["handle"]))
    return envelope


def _dict_to_ipc(d: dict[str, Any]) -> ipc.IpcHandle:
    return ipc.IpcHandle(
        mode=d["mode"],
        shape=tuple(d["shape"]),
        dtype=d["dtype"],
        device=d["device"],
        payload=d["payload"],
    )


def _config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if hasattr(config, "__dict__"):
        return {k: v for k, v in vars(config).items() if not k.startswith("_")}
    return dict(config)
