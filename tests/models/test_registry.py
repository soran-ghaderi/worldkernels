r"""Tests for the video-diffusion pipeline registry."""

from __future__ import annotations

import pytest

from worldkernels.models import get_pipeline_class, list_pipelines, register_pipeline


def test_wan_i2v_is_registered():
    assert "wan_i2v" in list_pipelines()


def test_get_pipeline_class_resolves_wan():
    cls = get_pipeline_class("wan_i2v")
    assert cls.__name__ == "WanI2VPipeline"


def test_get_pipeline_class_unknown_raises():
    with pytest.raises(KeyError, match="Unknown pipeline"):
        get_pipeline_class("does_not_exist")


def test_register_pipeline_adds_entry():
    register_pipeline("dummy_family", "worldkernels.models.wan.pipeline_wan_i2v", "WanLatent")
    assert "dummy_family" in list_pipelines()
    assert get_pipeline_class("dummy_family").__name__ == "WanLatent"
