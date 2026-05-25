r"""CUDA graph capture and replay.

A captured graph records the GPU kernel sequence of a fixed-shape callable
once, then replays it with near-zero CPU launch overhead — the largest win for
the denoise loop, which runs the same transformer step tens of times. Inputs
must keep identical shapes, dtypes, and devices across replays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch

__all__ = ["CUDAGraphRunner"]


class CUDAGraphRunner:
    r"""Captures one fixed-shape callable as a CUDA graph and replays it."""

    def __init__(self) -> None:
        self._graph: Any = None
        self._static_inputs: tuple["torch.Tensor", ...] = ()
        self._static_output: Any = None

    @property
    def is_captured(self) -> bool:
        return self._graph is not None

    def capture(
        self,
        fn: Callable[..., "torch.Tensor"],
        *sample_inputs: "torch.Tensor",
        warmup: int = 3,
    ) -> "CUDAGraphRunner":
        r"""Warm up and capture ``fn(*sample_inputs)`` as a CUDA graph.

        Args:
            fn: The callable to capture; must take only tensor arguments.
            sample_inputs: Representative inputs defining the captured shapes.
            warmup: Eager iterations before capture (lets autograd/cuDNN settle).
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graph capture requires a CUDA device")

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup):
                fn(*sample_inputs)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        self._static_inputs = tuple(t.clone() for t in sample_inputs)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = fn(*self._static_inputs)
        return self

    def replay(self, *inputs: "torch.Tensor") -> Any:
        r"""Copy ``inputs`` into the captured buffers and replay the graph."""
        if self._graph is None:
            raise RuntimeError("graph not captured; call capture() first")
        if len(inputs) != len(self._static_inputs):
            raise ValueError(f"expected {len(self._static_inputs)} inputs, got {len(inputs)}")
        for static, fresh in zip(self._static_inputs, inputs):
            static.copy_(fresh)
        self._graph.replay()
        return self._static_output
