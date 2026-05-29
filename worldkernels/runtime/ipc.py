r"""Tensor IPC across the engine ↔ worker boundary.

Two paths:
- **CUDA IPC**: shares the device pointer via ``cudaIpcGetMemHandle``.
  Zero-copy across processes on the same GPU.
- **Host shared memory**: ``multiprocessing.shared_memory`` with a D2H/H2D
  copy. Slower, used when CUDA IPC fails (cgroups, MIG, containers without
  ``--ipc=host``).

The owner side must keep the source tensor alive until the consumer releases
the handle (refcounting via the protocol header).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)

__all__ = ["IpcHandle", "share_tensor", "recv_tensor", "ipc_available"]

_DISABLE_CUDA_IPC: bool = False


def disable_cuda_ipc(disable: bool = True) -> None:
    r"""Force the host-shm fallback path (used by tests)."""
    global _DISABLE_CUDA_IPC
    _DISABLE_CUDA_IPC = disable


def ipc_available() -> bool:
    if _DISABLE_CUDA_IPC:
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@dataclass
class IpcHandle:
    r"""Serializable handle for a shared tensor.

    Either ``mode="cuda"`` with ``ipc_payload`` from ``Tensor._share_cuda_()``,
    or ``mode="shm"`` with the name of a ``multiprocessing.shared_memory`` block.
    """

    mode: Literal["cuda", "shm"]
    shape: tuple[int, ...]
    dtype: str
    device: str
    payload: dict[str, Any]


def share_tensor(t: Any) -> IpcHandle:
    if _DISABLE_CUDA_IPC or not _is_cuda_tensor(t):
        return _share_via_shm(t)
    try:
        return _share_via_cuda(t)
    except (RuntimeError, OSError) as exc:
        log.warning("CUDA IPC failed (%s); falling back to host shared memory", exc)
        return _share_via_shm(t)


def recv_tensor(handle: IpcHandle) -> Any:
    if handle.mode == "cuda":
        return _recv_via_cuda(handle)
    return _recv_via_shm(handle)


def _is_cuda_tensor(t: Any) -> bool:
    try:
        import torch

        return isinstance(t, torch.Tensor) and t.is_cuda
    except ImportError:
        return False


def _share_via_cuda(t: Any) -> IpcHandle:
    payload = t._share_cuda_()
    return IpcHandle(
        mode="cuda",
        shape=tuple(t.shape),
        dtype=str(t.dtype),
        device=str(t.device),
        payload={
            "device": payload[0],
            "handle": payload[1],
            "size_bytes": payload[2],
            "offset_bytes": payload[3],
            "ref_counter_handle": payload[4] if len(payload) > 4 else None,
            "ref_counter_offset": payload[5] if len(payload) > 5 else None,
            "event_handle": payload[6] if len(payload) > 6 else None,
            "event_sync_required": payload[7] if len(payload) > 7 else None,
        },
    )


def _recv_via_cuda(handle: IpcHandle) -> Any:
    import torch
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    p = handle.payload
    dtype = _str_to_dtype(handle.dtype)
    args = (
        torch.Tensor,
        handle.shape,
        (1,) * len(handle.shape),
        0,
        dtype,
        p["device"],
        p["handle"],
        p["size_bytes"],
        p["offset_bytes"],
        p.get("ref_counter_handle"),
        p.get("ref_counter_offset"),
        p.get("event_handle"),
        p.get("event_sync_required"),
    )
    return rebuild_cuda_tensor(*args)


def _share_via_shm(t: Any) -> IpcHandle:
    from multiprocessing import shared_memory

    arr = _tensor_to_numpy(t)
    shm = shared_memory.SharedMemory(create=True, size=max(1, arr.nbytes))
    shm.buf[: arr.nbytes] = arr.tobytes()
    return IpcHandle(
        mode="shm",
        shape=tuple(t.shape if hasattr(t, "shape") else arr.shape),
        dtype=str(getattr(t, "dtype", arr.dtype)),
        device=str(getattr(t, "device", "cpu")),
        payload={"name": shm.name, "size_bytes": arr.nbytes},
    )


def _recv_via_shm(handle: IpcHandle) -> Any:
    from multiprocessing import shared_memory

    import numpy as np

    shm = shared_memory.SharedMemory(name=handle.payload["name"])
    dtype = _str_to_numpy_dtype(handle.dtype)
    nbytes = handle.payload["size_bytes"]
    arr = np.frombuffer(shm.buf[:nbytes], dtype=dtype).reshape(handle.shape).copy()
    try:
        import torch

        out = torch.from_numpy(arr)
        target_device = handle.device
        if target_device.startswith("cuda"):
            out = out.to(target_device)
        return out
    except ImportError:
        return arr
    finally:
        shm.close()


def _tensor_to_numpy(t: Any):
    try:
        import torch

        if isinstance(t, torch.Tensor):
            return t.detach().contiguous().cpu().numpy()
    except ImportError:
        pass
    import numpy as np

    return np.asarray(t)


def _str_to_dtype(name: str):
    import torch

    table = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
    }
    if name in table:
        return table[name]
    short = name.split(".")[-1]
    return getattr(torch, short, torch.float32)


def _str_to_numpy_dtype(name: str):
    import numpy as np

    short = name.split(".")[-1]
    table = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": np.float32,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "bool": np.bool_,
    }
    return table.get(short, np.float32)
