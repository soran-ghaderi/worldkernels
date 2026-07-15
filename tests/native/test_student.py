r"""Replay the captured student streaming mechanism against the native pipeline.

The fixture was produced by the reference Self-Forcing rollout
(``generate_streaming_video``) bound to the causal reference net with real
teacher 2B weights (see `tests.fixtures.student_reference`); the native
`DreamDojoPipeline.generate_stream_latents` must reproduce its output latents.
Requires a GPU and the converted 2B_pretrain checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.native._assert import assert_close_to_reference, load_reference

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "data"
    / "student_mechanism_2b_240x320_seed7"
)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def student_fixture_dir() -> Path:
    if not FIXTURE_DIR.exists():
        pytest.skip(
            f"student fixture dir {FIXTURE_DIR} not found; run tests.fixtures.student_reference"
        )
    return FIXTURE_DIR


def test_stream_latents_replay_reference(student_fixture_dir: Path):
    import torch

    from worldkernels.models.dreamdojo.checkpoint import load_net
    from worldkernels.models.dreamdojo.pipeline import DreamDojoPipeline
    from worldkernels.runtime.forward_context import ForwardContext, set_forward_context

    manifest = json.loads((student_fixture_dir / "manifest.json").read_text())
    ckpt = manifest["ckpt"]
    if not Path(ckpt).exists():
        pytest.skip(f"checkpoint {ckpt} not cached")

    gt_latent = load_reference(student_fixture_dir, "gt_latent").cuda()
    init_noise = load_reference(student_fixture_dir, "init_noise").cuda()
    text_emb = load_reference(student_fixture_dir, "text_emb").cuda()
    action = load_reference(student_fixture_dir, "action").cuda()
    ref = load_reference(student_fixture_dir, "output_latents").cuda()

    pipeline = DreamDojoPipeline("2b_pretrain")
    pipeline.device, pipeline.dtype = "cuda", torch.bfloat16
    pipeline.net = load_net("2b_pretrain", ckpt_path=ckpt, device="cuda", dtype=torch.bfloat16)
    pipeline.text_emb = text_emb

    with set_forward_context(ForwardContext(attention_backend="sdpa")):
        out = pipeline.generate_stream_latents(
            gt_latent[:, :, : manifest["start_idx"]],
            action,
            init_noise,
            student_steps=manifest["n_steps"],
            window=manifest["cache_frame_size"],
        )
    assert_close_to_reference(
        out.float(), ref.float(), stage="pipeline_latent", name="student_stream"
    )
