---
sidebar_position: 8
title: Performance Optimization
description: Techniques for optimizing performance in WorldKernels
icon: material/speedometer
---

# Performance Optimization

WorldKernels is designed for GPU-first execution. This guide summarizes the optimization rules that matter most in practice when modifying runtime, adapters, or serving paths.

## Primary Performance Targets

Focus on these metrics when evaluating a change:

- step latency per session
- aggregate throughput under concurrent sessions
- VRAM usage stability over long runs
- cache hit behavior and scheduler efficiency

## Hot Path Priorities

Performance-critical modules include:

- `runtime/executor.py`
- `runtime/memory.py`
- `runtime/scheduler.py`
- `worlds/adapters/*` transition and decode calls

Prioritize these constraints:

1. No redundant allocations in per-step paths
2. No unnecessary copies between CPU and GPU
3. No repeated compute when state/cache reuse is possible

## Memory and Cache Practices

- Keep session state resident on GPU while active
- Reuse allocated buffers where shape constraints permit
- Treat latent and cache data as managed resources, not disposable tensors
- Apply explicit offload/eviction policies under VRAM pressure

A good default pattern is pre-allocation and reuse rather than creating tensors every step.

## Compute and Backend Strategy

- Prefer vectorized tensor operations over Python loops
- Use backend dispatch (`eager`, compiled backends) through runtime abstractions
- Warm up models/backends before measurement to remove one-time startup effects
- Keep dtype policy aligned with device capability (`bf16` on Ampere+, `fp16` fallback)

## Session Scheduling and Batching

Scheduler behavior has first-order impact on throughput:

- Batch only compatible sessions when doing so reduces amortized cost
- Preserve session-device affinity when state is already resident
- Use admission control to avoid over-committing VRAM
- Avoid policy changes that increase preemption churn without measured gain

## Serving-Layer Performance

- Keep API handlers non-blocking around I/O and orchestration
- Avoid expensive serialization work in tight streaming loops
- Decode only requested modalities to reduce unnecessary work

## Profiling Workflow

Use a layered profiling approach:

1. Reproduce with minimal workload
2. Profile runtime hotspots first
3. Validate end-to-end impact under concurrent sessions
4. Compare before/after with the same hardware and config

Useful commands:

```bash
pytest tests/ -x --tb=short
worldkernels bench latency --world dummy --steps 100
worldkernels bench throughput --world dummy --sessions 4
```

## Common Pitfalls

- Allocating tensors inside iterative transition loops
- Calling sync-heavy operations in hot paths
- Mixing concerns between scheduling and model compute logic
- Optimizing without representative benchmarks

## Performance Review Checklist

Before merging a performance-related change, confirm:

- Correctness is unchanged
- Memory behavior is stable for long sessions
- Measured metric improvements are reproducible
- Complexity increase is justified by meaningful gains
