r"""Tensor IPC: host-shm fallback path (CPU-only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from worldkernels.runtime import ipc


@pytest.fixture(autouse=True)
def _force_host_shm():
    ipc.disable_cuda_ipc(True)
    yield
    ipc.disable_cuda_ipc(False)


class TestHostShmFallback:
    def test_share_and_recv_cpu_tensor(self):
        src = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        handle = ipc.share_tensor(src)
        assert handle.mode == "shm"
        out = ipc.recv_tensor(handle)
        torch.testing.assert_close(out, src)

    def test_dtype_preserved(self):
        src = torch.ones((2, 2), dtype=torch.int32)
        handle = ipc.share_tensor(src)
        out = ipc.recv_tensor(handle)
        assert out.dtype == torch.int32
        torch.testing.assert_close(out, src)
