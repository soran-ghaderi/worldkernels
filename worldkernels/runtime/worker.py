r"""Worker subprocess entrypoint for isolated models (ADR-012 Tier 2).

Invoked as ``python -m worldkernels.runtime.worker --model {id} --socket {path}
--device {cuda:N}``. Runs inside the per-model isolated venv (`runtime/envs.py`).
Loads the world via the standard registry, listens on a Unix socket, dispatches
RPC ops, and shares tensors via `runtime/ipc.py` (CUDA IPC or host shm).
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldkernels.runtime import ipc
from worldkernels.runtime.rpc import RpcServer

log = logging.getLogger(__name__)


@dataclass
class WorkerState:
    world: Any = None
    device: str = "cuda"
    dtype: Any = None
    objects: dict[str, Any] = None

    def __post_init__(self) -> None:
        self.objects = {}


def _new_handle() -> str:
    return uuid.uuid4().hex


def _handle_init(state: WorkerState, args: dict[str, Any]) -> dict[str, Any]:

    from worldkernels.worlds.registry import get_world_class

    adapter = args["adapter"]
    kwargs = args.get("kwargs", {})
    device = args.get("device", "cuda")
    dtype_str = args.get("dtype", "torch.bfloat16")

    cls = get_world_class(adapter)
    dtype = _resolve_dtype(dtype_str)
    world = cls(**kwargs)
    world.initialize(device=device, dtype=dtype)

    state.world = world
    state.device = device
    state.dtype = dtype
    log.info("worker: loaded %s on %s (%s)", adapter, device, dtype)
    return {"adapter": adapter, "device": device, "dtype": str(dtype)}


def _handle_warmup(state: WorkerState, args: dict[str, Any]) -> dict[str, Any]:
    from worldkernels.config import WorldConfig

    cfg = WorldConfig(**args.get("config", {}))
    state.world.warmup(cfg)
    return {"warmed": True}


def _handle_profile_vram(state: WorkerState, args: dict[str, Any]) -> dict[str, Any]:
    from worldkernels.config import WorldConfig

    cfg = WorldConfig(**args.get("config", {}))
    return {"vram_mb": float(state.world.profile_vram(cfg))}


def _handle_create_initial_state(state: WorkerState, args: dict[str, Any]) -> dict[str, Any]:
    from worldkernels.config import WorldConfig

    cfg = WorldConfig(**args.get("config", {}))
    seed = int(args.get("seed", 0))
    latent_state = state.world.create_initial_state(cfg, seed)
    h = _new_handle()
    state.objects[h] = latent_state
    return {"handle": h}


def _handle_encode_action(state: WorkerState, args: dict[str, Any]) -> dict[str, Any]:
    from worldkernels.core.action import Action

    action = Action(action_type=args["action_type"], payload=args.get("payload", {}))
    tensor = state.world.encode_action(action)
    h = _new_handle()
    state.objects[h] = tensor
    return {"handle": h}


def _handle_transition(state: WorkerState, args: dict[str, Any]) -> dict[str, Any]:
    state_h = args["state_handle"]
    action_h = args["action_handle"]
    latent = state.objects[state_h]
    action = state.objects[action_h]
    new_state = state.world.transition(latent, action)
    new_h = _new_handle()
    state.objects[new_h] = new_state
    return {"handle": new_h}


def _handle_decode(state: WorkerState, args: dict[str, Any]) -> dict[str, Any]:
    state_h = args["state_handle"]
    modalities = args.get("modalities", ["frames"])
    latent = state.objects[state_h]
    t0 = time.perf_counter()
    obs = state.world.decode_observation(latent, modalities)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result: dict[str, Any] = {
        "step_index": obs.step_index,
        "generation_time_ms": getattr(obs, "generation_time_ms", elapsed_ms),
        "decode_skipped": getattr(obs, "decode_skipped", False),
    }
    if obs.frames is not None:
        result["frames"] = [_bytes_or_handle(f) for f in obs.frames]
    if getattr(obs, "latent", None) is not None:
        result["latent_ipc"] = _to_ipc(obs.latent)
    return result


def _bytes_or_handle(frame: Any) -> Any:
    if isinstance(frame, (bytes, bytearray)):
        return {"kind": "bytes", "data": bytes(frame)}
    handle = ipc.share_tensor(frame)
    return {"kind": "ipc", "handle": _ipc_to_dict(handle)}


def _to_ipc(t: Any) -> dict[str, Any]:
    return _ipc_to_dict(ipc.share_tensor(t))


def _ipc_to_dict(h: ipc.IpcHandle) -> dict[str, Any]:
    return {
        "mode": h.mode,
        "shape": list(h.shape),
        "dtype": h.dtype,
        "device": h.device,
        "payload": h.payload,
    }


def _handle_release(state: WorkerState, args: dict[str, Any]) -> dict[str, Any]:
    h = args["handle"]
    state.objects.pop(h, None)
    return {"released": h}


def _handle_health(state: WorkerState, args: dict[str, Any]) -> dict[str, Any]:
    import os

    return {
        "ok": True,
        "pid": os.getpid(),
        "device": state.device,
        "world_loaded": state.world is not None,
        "n_objects": len(state.objects or {}),
    }


def _handle_close(state: WorkerState, args: dict[str, Any]) -> dict[str, Any]:
    state.objects.clear()
    state.world = None
    return {"closed": True}


_DISPATCH = {
    "init": _handle_init,
    "warmup": _handle_warmup,
    "profile_vram": _handle_profile_vram,
    "create_initial_state": _handle_create_initial_state,
    "encode_action": _handle_encode_action,
    "transition": _handle_transition,
    "decode": _handle_decode,
    "release": _handle_release,
    "health": _handle_health,
    "close": _handle_close,
}


def _resolve_dtype(name: str) -> Any:
    import torch

    short = name.split(".")[-1]
    return getattr(torch, short, torch.float32)


def serve(socket_path: Path) -> None:
    state = WorkerState()

    def handler(op: str, args: dict[str, Any]) -> Any:
        fn = _DISPATCH.get(op)
        if fn is None:
            raise ValueError(f"unknown op: {op}")
        return fn(state, args)

    server = RpcServer(socket_path)
    log.info("worker: serving on %s", socket_path)
    server.serve(handler)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="worldkernels worker subprocess")
    p.add_argument("--socket", required=True, help="Unix socket path")
    p.add_argument("--log-level", default="INFO")
    ns = p.parse_args()

    logging.basicConfig(level=ns.log_level, format="worker[%(process)d] %(message)s")
    try:
        serve(Path(ns.socket))
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        log.exception("worker fatal")
        return 1


if __name__ == "__main__":
    sys.exit(main())
