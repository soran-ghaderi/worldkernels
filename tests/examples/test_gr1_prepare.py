r"""Lock the GR-1 action transform in examples/gr1_prepare.py to NVIDIA's groot_dreams recipe.

The model is conditioned on a 29-dim action = concat of the gr1 action keys (left_arm, right_arm,
left_hand, right_hand, waist) each ``min_max``-normalized to [-1, 1], then 4-step deltas. A wrong
column map or normalization mode produces out-of-distribution actions and gibberish video.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gr1_prepare", Path(__file__).resolve().parents[2] / "examples" / "gr1_prepare.py"
)
gp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gp)


def _write_meta(tmp: Path) -> Path:
    modality = {
        "action": {
            "left_arm": {"original_key": "action", "start": 964, "end": 971},
            "right_arm": {"original_key": "action", "start": 999, "end": 1006},
            "left_hand": {"original_key": "action", "start": 971, "end": 977},
            "right_hand": {"original_key": "action", "start": 1006, "end": 1012},
            "waist": {"original_key": "action", "start": 1027, "end": 1030},
            "absolute_left_hand": {"original_key": "action", "start": 0, "end": 96},
        }
    }
    (tmp / "modality.json").write_text(json.dumps(modality))
    return tmp / "modality.json"


def test_action_key_ranges_order_and_dims(tmp_path):
    ranges = gp._action_key_ranges(_write_meta(tmp_path))
    assert [k for k, _, _ in ranges] == gp._GR1_ACTION_KEYS
    assert ranges[0] == ("left_arm", 964, 971)
    assert sum(b - a for _, a, b in ranges) == gp._GR1_EEF_DIM == 29


def test_action_key_ranges_missing_key_raises(tmp_path):
    (tmp_path / "modality.json").write_text(
        json.dumps({"action": {"left_arm": {"start": 0, "end": 7}}})
    )
    with pytest.raises(SystemExit):
        gp._action_key_ranges(tmp_path / "modality.json")


def test_min_max_maps_extremes_to_pm1_and_degenerate_to_zero(tmp_path):
    lo = [0.0] * 1030
    hi = [0.0] * 1030
    for a, b in ((964, 971), (999, 1006), (971, 977), (1006, 1012), (1027, 1030)):
        for j in range(a, b):
            lo[j], hi[j] = -2.0, 2.0
    hi[1029] = -2.0  # waist last dim degenerate (min == max)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"action": {"min": lo, "max": hi}}))
    ranges = gp._action_key_ranges(_write_meta(tmp_path))

    action = np.zeros((3, 1030))
    action[0, :] = -2.0  # all at min -> -1
    action[1, :] = 2.0  # all at max -> +1
    action[2, :] = 0.0  # midpoint -> 0
    norm = gp._min_max_normalize(action, stats_path, ranges)

    assert norm.shape == (3, 29)
    assert np.allclose(norm[0, :28], -1.0)  # min -> -1 (exclude degenerate last)
    assert np.allclose(norm[1, :28], 1.0)  # max -> +1
    assert np.allclose(norm[2], 0.0)  # midpoint -> 0
    assert norm[0, 28] == 0.0  # min == max -> 0, not normalized


def test_delta_actions_formula():
    eef = np.arange(10 * 29, dtype=np.float64).reshape(10, 29)
    out = gp._delta_actions(eef)
    expected = np.concatenate([eef[t : t + 4] - eef[t - 1] for t in range(1, 9, 4)], axis=0)
    assert np.array_equal(out, expected)


def test_delta_actions_empty_short_sequence():
    assert gp._delta_actions(np.zeros((2, 29))).shape == (0, 29)
