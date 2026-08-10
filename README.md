# WorldKernels

**GPU-first world model simulation engine** — serve learned world models (DiT, VAE) as interactive sessions.

[![PyPI version](https://img.shields.io/pypi/v/worldkernels.svg)](https://pypi.org/project/worldkernels/)
[![Python](https://img.shields.io/pypi/pyversions/worldkernels.svg)](https://pypi.org/project/worldkernels/)
[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](https://www.gnu.org/licenses/lgpl-2.1)

WorldKernels manages latent state caches for iterative world simulation. Each world model is decomposed into independently schedulable pipeline stages (action encoding, state transition, observation decoding).

## Install

```bash
pip install worldkernels
```

That's it. On first use of a model, worldkernels lazily installs any extra runtime deps, clones any required github packages, and downloads weights from HuggingFace. You see one progress panel; you don't pick extras.

## Quick start

### CLI

`wk` is the short alias for `worldkernels` — both the command and the import
(`import wk` ≡ `import worldkernels`) work interchangeably.

```bash
wk serve nvidia/DreamDojo
wk run dreamdojo --steps 3 --prompt "robot picks up cube" -o out/
wk pull dreamdojo                 # pre-fetch without serving
wk models                         # list locally cached models
wk rm dreamdojo                   # free disk
wk collect-env                    # GPU, deps, hub, cache, isolated envs
```

The positional `<model>` accepts:
- a short alias (`dreamdojo`, `cosmos-predict2`, `dummy`)
- an HF repo id (`nvidia/DreamDojo`, `nvidia/Cosmos-Predict2.5-2B`)
- an HF URL (`https://huggingface.co/nvidia/DreamDojo`)
- a local checkpoint path (`./pretrain.pt`)

Variants are selected with `--variant`:
```bash
wk serve nvidia/DreamDojo --variant 2b_gr1
```

### Python API

```python
from worldkernels import WorldKernel, Action, WorldConfig

wk = WorldKernel(device="cuda")
wk.load_model("dreamdojo", variant="2b_gr1")

session = wk.create_session(
    "dreamdojo",
    config=WorldConfig(height=480, width=640, initial_prompt="A robot arm on a table"),
    seed=42,
)

for _ in range(10):
    obs = session.step(Action("joints", {"joints": [0.0] * 384}), modalities=["frames"])
```

### REST API

```bash
# Load (synchronous)
curl -X POST localhost:8000/v1/worlds \
  -H "Content-Type: application/json" \
  -d '{"model_id": "nvidia/DreamDojo", "variant": "2b_gr1"}'

# Load with streamed progress (SSE)
curl -N -X POST localhost:8000/v1/worlds:stream \
  -H "Content-Type: application/json" \
  -d '{"model_id": "nvidia/DreamDojo"}'

# Create session
curl -X POST localhost:8000/v1/sessions \
  -d '{"world": "dreamdojo", "height": 480, "width": 640}'

# Step
curl -X POST localhost:8000/v1/sessions/{id}/step \
  -d '{"action_type": "joints", "payload": {"joints": [0.0]}}'
```

## Supported models

| Model | Adapter | Status | VRAM |
|-------|---------|--------|------|
| DreamDojo 2B/14B (all variants) | `dreamdojo` | Validated | ~6–30 GB |
| Cosmos-Predict2.5-2B | `cosmos_predict2` | Validated | ~21 GB |
| Wan2.1 / Wan2.2 (I2V, TI2V) | `generator_world` | Implemented | varies |
| DummyWorld | `dummy` | Complete | 0 |

Gated HuggingFace repos (e.g. `nvidia/DreamDojo`) need `huggingface-cli login` first.

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

- **One-line install** — `pip install worldkernels` is enough; deps fetch lazily per model
- **Session-based API** — stateful simulation with checkpoint/branch/restore
- **Stage-decomposed pipeline** — encode, transition, decode independently schedulable
- **CLI** — `serve`, `run`, `pull`, `models`, `rm`, `collect-env`
- **GPU-optimized** — bf16, pre-allocated buffers, flash attention
- **HTTP server** — FastAPI REST API, with SSE progress streaming
- **Extensible** — plugin system via entry_points for custom world models

## Component toggles & ablation

Every worldkernels component is a `RuntimeConfig` field — you can flip any one on or off for benchmarking, demos, or A/B:

```bash
wk serve nvidia/DreamDojo --profile baseline      # every optimization off
wk serve nvidia/DreamDojo --profile production    # teacache + int8 quant
wk serve nvidia/DreamDojo --profile fast --set teacache=off,attention_backend=sdpa
wk bench latency --world dummy --profile baseline # ablation runs
wk config-show --profile production               # see every flag + source
WK_DISABLE=cuda_graphs,teacache wk serve ...      # env-var escape
```

Precedence is `CLI flag > env (WK_*) > profile > built-in default`. Per-session overrides for the safe subset (teacache, trajectory_cache, iteration_batching, offload_idle, attention_backend) ride `create_session(overrides={...})` or HTTP `POST /v1/sessions {"overrides": {...}}`. The full set of flags is at `worldkernels/config/runtime.py`.

## Power knobs

```bash
--variant NAME           # pick a non-default variant (e.g. 2b_gr1)
HF_HUB_ENABLE_HF_TRANSFER=0  # opt out of Rust-based parallel HF downloads (default on)
--ckpt-path PATH         # bypass HF download, point at a local file
--no-fetch               # error if anything is missing (CI / airgap)
-q / --quiet             # suppress progress output
WORLDKERNELS_HOME=...    # override cache root (default ~/.cache/worldkernels)
WORLDKERNELS_NO_AUTO_INSTALL=1   # same as --no-fetch globally
COSMOS_PREDICT2_PATH=... # use an existing DreamDojo checkout
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA (for non-dummy models)
- `huggingface-cli login` for gated repos

## Offline / CI install

If you can't fetch on first use, install all extras up front:

```bash
pip install "worldkernels[all]"
# or just the families you need:
pip install "worldkernels[cosmos]"      # transformers, diffusers, flash-attn, …
pip install "worldkernels[diffusion]"   # diffusers + Wan helpers
```

Then run with `--no-fetch` to fail fast on any unexpected fetch.

## License

LGPL-2.1 — see [LICENSE](LICENSE).
