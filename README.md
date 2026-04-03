# WorldKernels

**GPU-first world model simulation engine** — serve learned world models (DiT, VAE) as interactive sessions.

[![PyPI version](https://img.shields.io/pypi/v/worldkernels.svg)](https://pypi.org/project/worldkernels/)
[![Python](https://img.shields.io/pypi/pyversions/worldkernels.svg)](https://pypi.org/project/worldkernels/)
[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](https://www.gnu.org/licenses/lgpl-2.1)

Think "vLLM but for world models": where vLLM manages KV caches for autoregressive LLM inference, WorldKernels manages latent state caches for iterative world simulation. Each world model is decomposed into independently schedulable pipeline stages (action encoding, state transition, observation decoding).

## Installation

```bash
pip install worldkernels                # Core (torch, numpy, pydantic, tyro)
pip install worldkernels[serve]         # + FastAPI server
pip install worldkernels[cosmos]        # + Cosmos-Predict2.5 dependencies
pip install worldkernels[all]           # Everything
```

For Cosmos-Predict2.5-2B, you also need the `cosmos_predict2` package:

```bash
git clone https://github.com/NVIDIA/DreamDojo.git ~/DreamDojo
```

The adapter auto-discovers it from `~/DreamDojo` or `COSMOS_PREDICT2_PATH`. If git is available and the repo isn't found, it auto-clones on first use.

## Quick Start

### Python API

```python
from worldkernels import WorldKernel, Action, WorldConfig

wk = WorldKernel(device="cuda")
wk.load_model("cosmos_predict2", num_inference_steps=5, guidance_scale=7.0)

session = wk.create_session(
    "cosmos_predict2",
    config=WorldConfig(height=480, width=640, initial_prompt="A robot arm on a table"),
    seed=42,
)

for _ in range(10):
    obs = session.step(Action("text", {"prompt": ""}), modalities=["frames"])
    # obs.frames: list of raw RGB bytes per frame

ckpt = session.checkpoint()      # snapshot state
branch = session.branch()        # fork session (copy-on-write)
session.restore(ckpt)            # rollback

session.close()
wk.shutdown()
```

### CLI

```bash
# Serve with a pre-loaded model
worldkernels serve --model cosmos_predict2 --num-inference-steps 5

# Headless run
worldkernels run --world cosmos_predict2 --steps 3 --prompt "A city street" --output-dir ./frames

# System check
worldkernels doctor
```

### REST API

```bash
# Load model
curl -X POST localhost:8000/v1/worlds \
  -H "Content-Type: application/json" \
  -d '{"model_id": "cosmos_predict2", "kwargs": {"num_inference_steps": 5}}'

# Create session
curl -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"world": "cosmos_predict2", "height": 480, "width": 640}'

# Step
curl -X POST localhost:8000/v1/sessions/{id}/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "text", "payload": {"prompt": ""}}'
```

## Supported Models

| Model | Adapter | Status | VRAM | Notes |
|-------|---------|--------|------|-------|
| Cosmos-Predict2.5-2B | `cosmos_predict2` | Validated | ~21 GB | Auto-downloads from HuggingFace (gated, requires `huggingface-cli login`) |
| DreamDojo 2B/14B | `dreamdojo` | Implemented | ~6-30 GB | Action-conditioned (robot joints) |
| DummyWorld | `dummy` | Complete | ~0 | Random noise, for dev/testing |

## Architecture

Each world model is decomposed into three pipeline stages:

```
Action --> [Encode] --> [Transition] --> [Decode] --> Observation
            Stage 1      Stage 2        Stage 3
           (lightweight)  (DiT denoise)  (VAE decode)
```

Sessions are stateful GPU resources with checkpoint, branch, and restore:

```
Session A: step -> step -> checkpoint -> step -> step
                              |
                              +---> Branch B: step -> step (independent)
```

## Features

- **Session-based API** — stateful simulation with checkpoint/branch/restore
- **Stage-decomposed pipeline** — encode, transition, decode independently schedulable
- **Auto-download** — checkpoints fetched from HuggingFace on first use
- **GPU-optimized** — bf16, pre-allocated buffers, flash attention
- **HTTP server** — FastAPI REST API with model kwargs pass-through
- **CLI** — `serve`, `run`, `bench`, `doctor` commands with model-specific flags
- **Extensible** — plugin system via entry_points for custom world models

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- For Cosmos: `flash-attn`, HuggingFace auth for gated model

## Documentation

Full docs at [worldkernels.dev](https://github.com/soran-ghaderi/worldkernels)

## License

LGPL-2.1 — see [LICENSE](LICENSE)
