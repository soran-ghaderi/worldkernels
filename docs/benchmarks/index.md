---
icon: material/chart-line
---

# Benchmarks & Metrics

Auto-generated from `benchmarks/` scripts and `worldkernels/runtime/metrics.py`.

## Prometheus Metrics

The following metrics are exported on the `/metrics` endpoint.

| Metric Name | Type | Description |
| --- | --- | --- |
| `wk_frames_generated_total` | Counter :material-counter: | Total frames generated across all sessions. |
| `wk_steps_total` | Counter :material-counter: | Total simulation steps executed. |
| `wk_worker_respawns_total` | Counter :material-counter: | Times an isolated worker was respawned after crashing. |
| `wk_active_sessions` | Gauge :material-speedometer: | Number of live sessions. |
| `wk_cache_hit_ratio` | Gauge :material-speedometer: | Denoise-step cache hit ratio in [0, 1]. |
| `wk_model_isolation_tier` | Gauge :material-speedometer: | Per-model isolation tier (0 = shared env, 1 = isolated subprocess). |
| `wk_vram_usage_bytes` | Gauge :material-speedometer: | Device memory currently allocated. |
| `wk_worker_processes` | Gauge :material-speedometer: | Number of isolated worker subprocesses currently live. |
| `wk_batch_size` | Histogram :material-chart-bar: | Number of sessions per batched forward pass. |
| `wk_step_latency_seconds` | Histogram :material-chart-bar: | Wall-clock latency of one simulation step. |
| `wk_worker_ipc_latency_seconds` | Histogram :material-chart-bar: | Engine ↔ worker RPC round-trip latency. |
