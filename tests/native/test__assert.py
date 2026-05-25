r"""Tests for tests/native/_assert.py (the tolerance contract module itself)."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from tests.native._assert import (
    TOLERANCES,
    DriftReport,
    StageTolerance,
    assert_close_to_reference,
    load_reference,
)


class TestStageTolerance:
    def test_frozen(self):
        t = StageTolerance(1.0, 1e-2, "x")
        with pytest.raises(AttributeError):
            t.atol = 0.0


class TestToleranceTable:
    @pytest.mark.parametrize(
        "stage",
        [
            "text_embedding",
            "vae_encode",
            "vae_decode",
            "sampler_step",
            "dit_block",
            "dit_forward",
            "pipeline_latent",
            "pipeline_decoded_float",
            "pipeline_decoded_uint8",
        ],
    )
    def test_known_stages_present(self, stage):
        assert stage in TOLERANCES
        t = TOLERANCES[stage]
        assert t.atol >= 0
        assert t.rtol >= 0
        assert isinstance(t.description, str) and t.description


class TestAssertCloseToReference:
    def test_pass_exact(self):
        ref = torch.randn(4, 4)
        report = assert_close_to_reference(ref.clone(), ref, stage="sampler_step")
        assert report.passed is True
        assert report.max_abs == 0.0
        assert report.n_over_budget == 0

    def test_pass_within_budget(self):
        ref = torch.zeros(4, 4)
        out = ref + 1e-7
        r = assert_close_to_reference(out, ref, stage="sampler_step")
        assert r.passed is True

    def test_fail_over_budget(self):
        ref = torch.zeros(2, 2)
        out = ref + 1.0
        with pytest.raises(AssertionError, match="FAIL"):
            assert_close_to_reference(out, ref, stage="sampler_step")

    def test_unknown_stage(self):
        with pytest.raises(KeyError, match="unknown tolerance stage"):
            assert_close_to_reference(torch.zeros(1), torch.zeros(1), stage="zzz")

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError, match="shape mismatch"):
            assert_close_to_reference(torch.zeros(3), torch.zeros(4), stage="sampler_step")

    def test_drift_report_string(self):
        ref = torch.zeros(2, 2)
        out = torch.zeros(2, 2)
        report = assert_close_to_reference(out, ref, stage="vae_encode")
        s = str(report)
        assert "PASS" in s
        assert "vae_encode" in s

    def test_named_report(self):
        ref = torch.zeros(2, 2)
        out = torch.zeros(2, 2)
        report = assert_close_to_reference(out, ref, stage="vae_encode", name="block0")
        assert report.name == "block0"
        assert "block0" in str(report)

    def test_uint8_tolerance(self):
        ref = torch.zeros(4, 4, dtype=torch.uint8)
        out = ref.clone() + 1
        r = assert_close_to_reference(out, ref, stage="pipeline_decoded_uint8")
        assert r.passed is True

    def test_uint8_overshoot(self):
        ref = torch.zeros(4, 4, dtype=torch.uint8)
        out = ref.clone() + 5
        with pytest.raises(AssertionError):
            assert_close_to_reference(out, ref, stage="pipeline_decoded_uint8")


class TestLoadReference:
    def test_loads_tensor(self, tmp_path):
        t = torch.randn(3, 5)
        save_file({"tensor": t}, str(tmp_path / "vae.safetensors"))
        loaded = load_reference(tmp_path, "vae")
        assert torch.equal(loaded, t)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_reference(tmp_path, "ghost")

    def test_missing_tensor_key(self, tmp_path):
        save_file({"other": torch.zeros(1)}, str(tmp_path / "bad.safetensors"))
        with pytest.raises(KeyError, match="missing 'tensor' key"):
            load_reference(tmp_path, "bad")


class TestDriftReportFields:
    def test_all_fields_populated(self):
        ref = torch.ones(2, 3)
        out = ref + 1e-9
        r = assert_close_to_reference(out, ref, stage="sampler_step", name="dummy")
        assert r.ref_shape == (2, 3)
        assert r.out_shape == (2, 3)
        assert "float32" in r.ref_dtype
        assert r.n_total == 6
        assert isinstance(r, DriftReport)
