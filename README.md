# WorldKernels

**GPU-first world model simulation engine** — serve learned world models (DiT, VAE) as interactive sessions.

[![PyPI version](https://img.shields.io/pypi/v/worldkernels.svg)](https://pypi.org/project/worldkernels/)
[![Python](https://img.shields.io/pypi/pyversions/worldkernels.svg)](https://pypi.org/project/worldkernels/)
[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](https://www.gnu.org/licenses/lgpl-2.1)

> ⚠️ **Early Development** — This package is in pre-alpha. APIs may change.

## Installation

```bash
pip install worldkernels
```

For full functionality:

```bash
pip install worldkernels[all]       # Everything
pip install worldkernels[serve]     # HTTP/WebSocket server
pip install worldkernels[diffusers] # HuggingFace Diffusers support
```

## Quick Start

```python
from worldkernels import WorldKernel, Action, WorldConfig

# Initialize engine
wk = WorldKernel(device="cuda")

# Load a world model from HuggingFace Hub
wk.load_world("Etched/oasis-500m")

# Create an interactive session
session = wk.create_session(
    world="oasis-500m",
    config=WorldConfig(height=360, width=640, fps=20),
)

# Step through the simulation
for _ in range(100):
    action = Action("keyboard", {"keys": ["W", "SPACE"]})
    obs = session.step(action)
    # obs.frames contains generated video frames

session.close()
wk.shutdown()
```

## Features (Planned)

- 🎮 **Session-based API** — Stateful simulation with checkpoint/branch
- 🚀 **GPU-optimized** — Pre-allocated buffers, CUDA graphs, torch.compile
- 🔌 **HuggingFace native** — Load models directly from the Hub
- 🌐 **HTTP/WebSocket server** — REST API and real-time streaming
- 🧩 **Extensible backends** — PyTorch eager, torch.compile, TensorRT

## Documentation

Coming soon at [worldkernels.dev](https://github.com/soran-ghaderi/worldkernels)

## License

LGPL-2.1 — see [LICENSE](LICENSE)
