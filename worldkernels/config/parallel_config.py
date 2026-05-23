r"""Parallelism configuration for distributed execution."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ParallelConfig"]


@dataclass
class ParallelConfig:
    r"""Multi-GPU parallelism degrees.

    The total rank count is the product of every degree,
    \( W = \text{TP} \cdot \text{DP} \cdot \text{PP} \cdot \text{SP} \cdot \text{CFG} \),
    where \( \text{SP} = \text{ulysses} \cdot \text{ring} \). Single-GPU
    execution is the all-ones special case.

    Args:
        tensor_parallel_size: Shards model weights across ranks.
        data_parallel_size: Replicates the model across rank groups.
        pipeline_parallel_size: Splits layers across rank stages.
        ulysses_degree: Ulysses sequence-parallel degree (all-to-all on Q/K/V).
        ring_degree: Ring sequence-parallel degree (ring attention).
        cfg_parallel_size: 1, or 2 to split positive/negative guidance.
    """

    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    ulysses_degree: int = 1
    ring_degree: int = 1
    cfg_parallel_size: int = 1

    def __post_init__(self) -> None:
        degrees = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "ulysses_degree": self.ulysses_degree,
            "ring_degree": self.ring_degree,
        }
        for name, value in degrees.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.cfg_parallel_size not in (1, 2):
            raise ValueError(f"cfg_parallel_size must be 1 or 2, got {self.cfg_parallel_size}")

    @property
    def sequence_parallel_size(self) -> int:
        return self.ulysses_degree * self.ring_degree

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel_size
            * self.data_parallel_size
            * self.pipeline_parallel_size
            * self.sequence_parallel_size
            * self.cfg_parallel_size
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1
