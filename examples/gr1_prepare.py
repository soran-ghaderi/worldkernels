r"""Prepare a DreamDojo GR-1 conditioning chunk from a PhysicalAI-Robotics-GR00T-Teleop-GR1 episode.

Turns one LeRobot episode (an ego-view ``.mp4`` + a ``.parquet`` of robot states/actions) into the
two inputs the action-conditioned ``dreamdojo --variant 2b_gr1`` model expects:

  - ``gr1_first_frame.png``  : the real first RGB frame, resized to the GR-1 native 480x640.
  - ``gr1_actions.npy``      : an ``[N, 384]`` action sequence; for GR-1 only ``[:, :29]`` is filled
                               (4-step end-effector deltas), the rest zero, per groot_dreams
                               ``dataset.py`` ``action_seq[:, :29] = delta_actions``.

Then run, e.g.:

```bash
worldkernels run dreamdojo --variant 2b_gr1 --height 480 --width 640 --steps 6 \
  -i datasets/gr1_cucumber/gr1_first_frame.png \
  --actions datasets/gr1_cucumber/gr1_actions.npy \
  --profile fast --set attention_backend=sdpa -o ./out/gr1/
```

The 29-dim action reproduces NVIDIA's groot_dreams transform for the ``gr1`` embodiment:
concatenate the action modality keys in config order (``left_arm``, ``right_arm``, ``left_hand``,
``right_hand``, ``waist`` = 7+7+6+6+3), each normalized to \([-1, 1]\) by per-key ``min_max`` from
the dataset ``stats.json``, sampled at ``timestep_interval=2``, then 4-step deltas
``action[t:t+4] - action[t-1]`` placed in ``action_seq[:, :29]``. Column ranges are read from the
dataset's own ``meta/modality.json`` (not hardcoded). The only un-modeled detail is the per-sample
base-index/horizon windowing (here the whole episode is processed as one stream); validate on GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Action modality keys for the gr1 embodiment, in concat order (groot_configs.py
# construct_modality_config_and_transforms): left_arm(7)+right_arm(7)+left_hand(6)+right_hand(6)
# +waist(3) = 29. Column ranges are resolved per-dataset from meta/modality.json.
_GR1_ACTION_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
_TIMESTEP_INTERVAL = 2
_ACTION_DIM = 384
_GR1_EEF_DIM = 29
_CHUNK = 12


def _read_parquet(path: Path) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - user env hint
        raise SystemExit("need pyarrow: `uv pip install pyarrow`") from exc
    col = pq.read_table(path, columns=["action"]).column("action").to_pylist()
    return np.asarray(col, dtype=np.float64)  # [T, 1030]


def _action_key_ranges(modality_path: Path) -> list[tuple[str, int, int]]:
    action_map = json.loads(modality_path.read_text())["action"]
    ranges = []
    for key in _GR1_ACTION_KEYS:
        if key not in action_map:
            raise SystemExit(f"action key {key!r} missing from {modality_path}")
        entry = action_map[key]
        ranges.append((key, int(entry["start"]), int(entry["end"])))
    return ranges


def _min_max_normalize(
    action: np.ndarray, stats_path: Path, ranges: list[tuple[str, int, int]]
) -> np.ndarray:
    r"""Per-key ``min_max`` to \([-1, 1]\): \(2 (x - \min)/(\max - \min) - 1\), 0 where min==max."""
    stats = json.loads(stats_path.read_text())["action"]
    lo = np.asarray(stats["min"], dtype=np.float64)
    hi = np.asarray(stats["max"], dtype=np.float64)
    cols = []
    for _, a, b in ranges:
        x = action[:, a:b]
        span = hi[a:b] - lo[a:b]
        mask = span != 0
        norm = np.zeros_like(x)
        norm[:, mask] = 2.0 * (x[:, mask] - lo[a:b][mask]) / span[mask] - 1.0
        cols.append(norm)
    return np.concatenate(cols, axis=1)  # [T, 29]


def _delta_actions(eef: np.ndarray) -> np.ndarray:
    r"""4-step eef deltas, matching groot_dreams dataset.py: action[t:t+4] - action[t-1]."""
    out = [eef[t : t + 4] - eef[t - 1] for t in range(1, len(eef) - 1, 4)]
    return np.concatenate(out, axis=0) if out else np.zeros((0, _GR1_EEF_DIM))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", required=True, help="…/In-lab_Eval/gr1_unified.<task>_robot")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--camera", default="observation.images.ego_view_freq20")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument(
        "--zero-actions", action="store_true", help="emit static (zero) actions baseline"
    )
    args = ap.parse_args()

    task = Path(args.task_dir).expanduser()
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ep = f"episode_{args.episode:06d}"

    import imageio.v3 as iio

    mp4 = task / "videos" / "chunk-000" / args.camera / f"{ep}.mp4"
    frame = iio.imread(mp4, index=0, plugin="pyav")  # [H,W,3] uint8
    try:
        from PIL import Image

        frame = np.asarray(Image.fromarray(frame).resize((args.width, args.height)))
    except ImportError:
        pass
    frame_path = out / "gr1_first_frame.png"
    iio.imwrite(frame_path, frame)

    n_steps = 0
    if args.zero_actions:
        action_seq = np.zeros((_CHUNK, _ACTION_DIM), dtype=np.float32)
    else:
        action_1030 = _read_parquet(task / "data" / "chunk-000" / f"{ep}.parquet")
        action_1030 = action_1030[::_TIMESTEP_INTERVAL]
        ranges = _action_key_ranges(task / "meta" / "modality.json")
        norm = _min_max_normalize(action_1030, task / "meta" / "stats.json", ranges)
        deltas = _delta_actions(norm).astype(np.float32)  # [N, 29]
        action_seq = np.zeros((len(deltas), _ACTION_DIM), dtype=np.float32)
        action_seq[:, :_GR1_EEF_DIM] = deltas
        n_steps = int(np.ceil(len(deltas) / _CHUNK))
    actions_path = out / "gr1_actions.npy"
    np.save(actions_path, action_seq)

    print(f"first frame : {frame_path}  ({args.height}x{args.width})")
    print(f"actions     : {actions_path}  shape {action_seq.shape}  (~{n_steps or 1} chunks)")
    print("\nrun:")
    print(
        f"  worldkernels run dreamdojo --variant 2b_gr1 --height {args.height} "
        f"--width {args.width} --steps {max(n_steps, 1)} -i {frame_path} "
        f"--actions {actions_path} --profile fast --set attention_backend=sdpa -o ./out/gr1/"
    )


if __name__ == "__main__":
    main()
