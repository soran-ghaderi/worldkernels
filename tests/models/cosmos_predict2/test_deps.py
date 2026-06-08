r"""Tests for worldkernels/worlds/pipelines/cosmos_predict2/deps.py."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from worldkernels.models.cosmos_predict2 import deps


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    snapshot = dict(sys.modules)
    meta_path = list(sys.meta_path)
    setup = deps._setup_done
    yield
    deps._setup_done = setup
    sys.meta_path[:] = meta_path
    for key in list(sys.modules):
        if key not in snapshot:
            del sys.modules[key]


class TestQuietVendorTqdm:
    def test_defaults_leave_false(self, monkeypatch):
        fake = types.ModuleType("tqdm")

        class _T:
            def __init__(self, *a, **k):
                self.kw = k

        fake.tqdm = _T
        monkeypatch.setitem(sys.modules, "tqdm", fake)
        deps._quiet_vendor_tqdm()
        import tqdm

        assert tqdm.tqdm([1], desc="Generating samples").kw.get("leave") is False

    def test_respects_explicit_leave_and_is_idempotent(self, monkeypatch):
        fake = types.ModuleType("tqdm")

        class _T:
            def __init__(self, *a, **k):
                self.kw = k

        fake.tqdm = _T
        monkeypatch.setitem(sys.modules, "tqdm", fake)
        deps._quiet_vendor_tqdm()
        deps._quiet_vendor_tqdm()
        import tqdm

        assert tqdm.tqdm([1], leave=True).kw.get("leave") is True
        assert getattr(tqdm.tqdm, "_wk_leave_false", False) is True


class TestInjectStub:
    def test_creates_module_with_attrs(self):
        deps._inject_stub("_pytest_stub_module", {"foo": 1, "bar": "x"})
        m = sys.modules["_pytest_stub_module"]
        assert isinstance(m, types.ModuleType)
        assert m.foo == 1
        assert m.bar == "x"

    def test_already_present_returns_existing(self):
        marker = types.ModuleType("_pytest_existing")
        marker.tag = "kept"
        sys.modules["_pytest_existing"] = marker
        out = deps._inject_stub("_pytest_existing", {"foo": 1})
        assert out is marker
        assert not hasattr(out, "foo")

    def test_dotted_module_attaches_to_parent(self):
        deps._inject_stub("_pytest_parent")
        deps._inject_stub("_pytest_parent.child", {"k": "v"})
        parent = sys.modules["_pytest_parent"]
        assert hasattr(parent, "child")
        assert parent.child.k == "v"


class TestInjectTEStubs:
    def test_creates_transformer_engine_stubs(self):
        deps._inject_te_stubs()
        assert "transformer_engine" in sys.modules
        assert "transformer_engine.pytorch" in sys.modules
        assert "transformer_engine.pytorch.attention" in sys.modules
        assert "transformer_engine.pytorch.attention.rope" in sys.modules

    def test_rmsnorm_normalizes(self):
        deps._inject_te_stubs()
        from transformer_engine.pytorch import RMSNorm

        norm = RMSNorm(8)
        x = torch.randn(2, 4, 8) * 5
        out = norm(x)
        rms = out.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)

    def test_rmsnorm_reset_parameters(self):
        deps._inject_te_stubs()
        from transformer_engine.pytorch import RMSNorm

        norm = RMSNorm(4)
        with torch.no_grad():
            norm.weight.zero_()
        norm.reset_parameters()
        assert torch.all(norm.weight == 1.0)

    def test_apply_rotary_pos_emb_preserves_shape(self):
        deps._inject_te_stubs()
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb

        t = torch.randn(4, 2, 1, 8)
        freqs = torch.randn(4, 1, 1, 8)
        out = apply_rotary_pos_emb(t, freqs)
        assert out.shape == t.shape

    def test_apply_rotary_bshd_format_transposes(self):
        r"""bshd path with freqs.shape[1]==1 hits the transpose branch."""
        deps._inject_te_stubs()
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb

        t = torch.randn(2, 4, 1, 8)
        freqs = torch.randn(4, 1, 1, 8)
        out = apply_rotary_pos_emb(t, freqs, tensor_format="bshd")
        assert out.shape == t.shape

    def test_dot_product_attention_4d_input_falls_back(self):
        deps._inject_te_stubs()
        from transformer_engine.pytorch.attention import DotProductAttention

        attn = DotProductAttention(num_attention_heads=2, kv_channels=8)
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        out = attn(q, k, v)
        assert out.shape == (1, 2, 4, 8)

    def test_dot_product_attention_3d_input(self):
        deps._inject_te_stubs()
        from transformer_engine.pytorch.attention import DotProductAttention

        attn = DotProductAttention(num_attention_heads=2, kv_channels=8)
        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        out = attn(q, k, v, attn_mask_type="causal")
        assert out.shape == (2, 4, 8)


class TestInjectTrainingStubs:
    def test_skips_when_megatron_already_present(self):
        sys.modules["megatron"] = types.ModuleType("megatron")
        deps._inject_training_stubs()
        m = sys.modules["megatron"]
        assert not hasattr(m, "core")

    def test_injects_full_set(self):
        for k in ("megatron", "transformer_engine", "transformer_engine_torch", "pytorch3d"):
            sys.modules.pop(k, None)
        deps._inject_training_stubs()
        assert "megatron" in sys.modules
        assert "megatron.core" in sys.modules
        assert "megatron.core.parallel_state" in sys.modules
        assert "transformer_engine" in sys.modules
        assert "transformer_engine_torch" in sys.modules
        assert "pytorch3d" in sys.modules
        assert "pytorch3d.transforms" in sys.modules

        import pytorch3d.transforms as t3

        m = torch.eye(3).unsqueeze(0)
        assert t3.matrix_to_rotation_6d(m).shape == (1, 6)
        assert t3.rotation_6d_to_matrix(torch.zeros(1, 6)).shape == (1, 3, 3)


class TestNoOpTrainingStubs:
    def test_noop_module_is_falsy_and_swallows_access(self):
        m = deps._NoOpModule("x")
        assert bool(m) is False
        assert m.anything is not None
        assert m.a.b.c is not None
        assert m(1, 2, k=3) is not None
        assert list(m) == []
        with m as ctx:
            assert ctx is not None

    def test_noop_attr_is_subclassable(self, monkeypatch):
        r"""A no-op'd lib used as a base class (e.g. webdataset.WebLoader) must be subclassable."""
        name = "_pytest_absent_baseclass_lib"
        monkeypatch.setattr(deps, "_TRAINING_ONLY", frozenset({name}))
        sys.modules.pop(name, None)
        deps._install_training_stub_finder()

        import importlib

        mod = importlib.import_module(name)

        class Sub(mod.WebLoader):  # was: TypeError module() takes at most 2 arguments
            pass

        assert Sub() is not None
        assert bool(mod.run) is False  # falsy guard (if wandb.run:) preserved
        assert mod.log({"loss": 1}) is not None  # callable preserved
        assert mod.AlertLevel.ERROR is not None  # nested attribute access preserved

    def test_noop_attrs_support_multiple_inheritance(self, monkeypatch):
        r"""Two stub symbols as bases must stay distinct: cosmos does
        ``class WebDataset(DataPipeline, FluidInterface)`` where both resolve through a no-op'd
        ``webdataset`` — a shared `_NoOp` singleton would raise ``duplicate base class``."""
        name = "_pytest_absent_mi_lib"
        monkeypatch.setattr(deps, "_TRAINING_ONLY", frozenset({name}))
        sys.modules.pop(name, None)
        deps._install_training_stub_finder()

        import importlib

        compat = importlib.import_module(f"{name}.compat")
        pipeline = importlib.import_module(f"{name}.pipeline")
        FluidInterface = compat.FluidInterface
        DataPipeline = pipeline.DataPipeline

        assert DataPipeline is not FluidInterface
        assert issubclass(DataPipeline, deps._NoOp)
        assert issubclass(FluidInterface, deps._NoOp)
        assert compat.FluidInterface is FluidInterface  # cached: identity-stable on repeat access
        assert pipeline.DataPipeline is DataPipeline

        class WebDataset(DataPipeline, FluidInterface):
            pass

        assert issubclass(WebDataset, deps._NoOp)
        assert bool(WebDataset()) is False  # value semantics preserved
        assert WebDataset()(1, 2) is not None

    def test_finder_stubs_absent_allowlisted(self, monkeypatch):
        name = "_pytest_absent_training_lib"
        monkeypatch.setattr(deps, "_TRAINING_ONLY", frozenset({name}))
        sys.modules.pop(name, None)
        deps._install_training_stub_finder()

        import importlib

        mod = importlib.import_module(name)
        assert isinstance(mod, deps._NoOpModule)
        assert bool(mod.run) is False
        assert mod.log({"loss": 1}) is not None
        assert mod.AlertLevel.ERROR is not None

    def test_finder_handles_deep_submodule(self, monkeypatch):
        r"""The botocore.config / multistorageclient.types case."""
        monkeypatch.setattr(deps, "_TRAINING_ONLY", frozenset({"_pytest_fake_storage"}))
        deps._install_training_stub_finder()
        mod = __import__("_pytest_fake_storage.config", fromlist=["Thing"])
        assert isinstance(mod, deps._NoOpModule)
        assert mod.Thing is not None

    def test_finder_does_not_shadow_installed(self, monkeypatch):
        monkeypatch.setattr(deps, "_TRAINING_ONLY", frozenset({"colorsys"}))
        sys.modules.pop("colorsys", None)
        deps._install_training_stub_finder()
        import colorsys

        assert not isinstance(colorsys, deps._NoOpModule)
        assert colorsys.rgb_to_hls(0.0, 0.0, 0.0) is not None

    def test_finder_non_allowlisted_still_raises(self, monkeypatch):
        monkeypatch.setattr(deps, "_TRAINING_ONLY", frozenset({"wandb"}))
        deps._install_training_stub_finder()
        with pytest.raises(ModuleNotFoundError):
            import _pytest_definitely_absent_xyz123  # noqa: F401

    def test_finder_idempotent(self):
        deps._install_training_stub_finder()
        n = sum(isinstance(f, deps._TrainingStubFinder) for f in sys.meta_path)
        deps._install_training_stub_finder()
        assert sum(isinstance(f, deps._TrainingStubFinder) for f in sys.meta_path) == n == 1

    def test_allowlist_covers_known_blockers(self):
        for name in ("wandb", "pynvml", "botocore", "multistorageclient", "albumentations", "av"):
            assert name in deps._TRAINING_ONLY

    def test_allowlist_covers_text_preprocessing(self):
        for name in ("ftfy", "nltk"):
            assert name in deps._TRAINING_ONLY

    def test_allowlist_excludes_guarded_compute_libs(self):
        r"""No-op'ing these would defeat cosmos's own try/except fallback or corrupt compute."""
        excluded = ("natten", "flash_attn", "flash_attn_3", "cudnn", "triton", "iopath", "imageio")
        for name in excluded:
            assert name not in deps._TRAINING_ONLY

    def test_flash_attn_shim_importable_but_raises(self):
        deps._install_flash_attn_shim()
        import flash_attn
        from flash_attn.layers.rotary import apply_rotary_emb

        assert isinstance(flash_attn, deps._FlashAttnStubModule)
        with pytest.raises(RuntimeError, match="flash_attn is not installed"):
            apply_rotary_emb(1, 2)
        with pytest.raises(RuntimeError, match="flash_attn is not installed"):
            flash_attn.flash_attn_varlen_func(1, 2, 3)

    def test_reason1_transformers_compat_backfills_default_rope(self):
        r"""reason1 looks up ROPE_INIT_FUNCTIONS['default'] (the 4.51-era key dropped in 5.x)."""
        pytest.importorskip("transformers")
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        had_default = "default" in ROPE_INIT_FUNCTIONS
        try:
            deps.install_reason1_transformers_compat()
            assert "default" in ROPE_INIT_FUNCTIONS
            deps.install_reason1_transformers_compat()  # idempotent
            assert "default" in ROPE_INIT_FUNCTIONS
        finally:
            if not had_default:
                ROPE_INIT_FUNCTIONS.pop("default", None)

    def test_lightning_module_is_subclassable_nn_module(self):
        r"""LightningModule is a base class for instantiated components, so it must be a real
        nn.Module — a no-op module is unsubclassable (the LAM(LightningModule) crash)."""
        import torch.nn as nn

        deps._inject_training_stubs()
        from lightning import LightningModule

        class _LAM(LightningModule):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 2)

        m = _LAM()
        assert isinstance(m, nn.Module)
        assert list(m.parameters())  # submodule registered
        m.log("x", 1)  # lightning-only API is a no-op
        assert m.global_step == 0

    def test_lightning_submodules_and_dummies_importable(self):
        deps._inject_training_stubs()
        from lightning import LightningDataModule
        from lightning.pytorch.cli import LightningCLI

        class _DM(LightningDataModule):  # dummy base, subclassable
            pass

        assert _DM() is not None
        assert LightningCLI is not None

    def test_lightning_not_in_noop_allowlist(self):
        assert "lightning" not in deps._TRAINING_ONLY
        assert "pytorch_lightning" not in deps._TRAINING_ONLY

    def test_flash_attn_shim_registers_distribution(self):
        r"""flash_attn must look like a real distribution so transformers' detection works."""
        import importlib.metadata as md

        deps._install_flash_attn_shim()
        assert md.version("flash_attn") == "2.8.0"
        pd = md.packages_distributions()
        assert "flash_attn" in pd  # the transformers PACKAGE_DISTRIBUTION_MAPPING KeyError site
        assert [p.replace("_", "-") for p in pd["flash_attn"]] == ["flash-attn"]

    def test_flash_attn_3_left_absent(self):
        deps._install_flash_attn_shim()
        with pytest.raises(ModuleNotFoundError):
            import flash_attn_3  # noqa: F401

    def test_flash_attn_shim_idempotent(self):
        deps._install_flash_attn_shim()
        n = sum(isinstance(f, deps._FlashAttnStubFinder) for f in sys.meta_path)
        deps._install_flash_attn_shim()
        assert sum(isinstance(f, deps._FlashAttnStubFinder) for f in sys.meta_path) == n == 1

    def test_training_stubs_installs_finder(self, monkeypatch):
        synth = "_pytest_absent_train_via_full"
        monkeypatch.setattr(deps, "_TRAINING_ONLY", frozenset({synth}))
        sys.modules.pop("megatron", None)
        deps._inject_training_stubs()
        assert any(isinstance(f, deps._TrainingStubFinder) for f in sys.meta_path)
        import importlib

        assert isinstance(importlib.import_module(synth), deps._NoOpModule)


class TestFindCosmosPredict2:
    def test_env_var_takes_precedence(self, monkeypatch, tmp_path):
        root = tmp_path / "repo"
        (root / "cosmos_predict2").mkdir(parents=True)
        (root / "cosmos_predict2" / "__init__.py").write_text("")
        monkeypatch.setenv("COSMOS_PREDICT2_PATH", str(root))
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "alt_home")
        assert deps._find_cosmos_predict2() == str(root)

    def test_no_paths_return_none(self, monkeypatch, tmp_path):
        r"""Force every candidate to be a fresh empty dir, ensuring None."""
        monkeypatch.setattr(
            "os.environ.get",
            lambda key, default="": (
                str(tmp_path / "empty_env") if key == "COSMOS_PREDICT2_PATH" else default
            ),  # noqa: E501
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "empty_home")
        monkeypatch.setattr(
            "pathlib.Path.resolve",
            lambda self, **kw: Path(str(tmp_path / "nonexistent_project")),
        )
        assert deps._find_cosmos_predict2() is None

    def test_finds_in_home_dreamdojo(self, monkeypatch, tmp_path):
        monkeypatch.delenv("COSMOS_PREDICT2_PATH", raising=False)
        home = tmp_path
        monkeypatch.setattr(Path, "home", lambda: home)
        dd = home / "DreamDojo"
        (dd / "cosmos_predict2").mkdir(parents=True)
        (dd / "cosmos_predict2" / "__init__.py").write_text("")
        monkeypatch.setattr(
            "pathlib.Path.resolve",
            lambda self, **kw: Path(str(tmp_path / "nonexistent_project")),
        )
        assert deps._find_cosmos_predict2() == str(dd)


class TestTryCloneCosmosPredict2:
    def test_no_git_returns_none(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert deps._try_clone_cosmos_predict2() is None

    def test_existing_target_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/git")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / "DreamDojo").mkdir()
        assert deps._try_clone_cosmos_predict2() is None

    def test_clone_success(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/git")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        def fake_run(cmd, **_kwargs):
            dest = Path(cmd[-1])
            (dest / "cosmos_predict2").mkdir(parents=True)
            (dest / "cosmos_predict2" / "__init__.py").write_text("")
            return MagicMock(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        result = deps._try_clone_cosmos_predict2()
        assert result == str(tmp_path / "DreamDojo")

    def test_clone_exception_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/git")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("subprocess.run", MagicMock(side_effect=RuntimeError("boom")))
        assert deps._try_clone_cosmos_predict2() is None


class TestEnsureCosmosPredict2:
    def test_idempotent_after_first_run(self, monkeypatch):
        deps._setup_done = True
        called = []
        monkeypatch.setattr(deps, "_inject_training_stubs", lambda: called.append("stubs"))
        deps.ensure_cosmos_predict2()
        assert called == []

    def test_existing_import_short_circuits_search(self, monkeypatch):
        deps._setup_done = False
        sys.modules["cosmos_predict2"] = types.ModuleType("cosmos_predict2")
        monkeypatch.setattr(
            deps, "_find_cosmos_predict2", lambda: pytest.fail("must not be called")
        )  # noqa: E501
        deps.ensure_cosmos_predict2()
        assert deps._setup_done is True

    def test_missing_with_no_repo_and_no_clone_raises(self, monkeypatch):
        deps._setup_done = False
        sys.modules.pop("cosmos_predict2", None)
        monkeypatch.setattr(
            "importlib.import_module",
            MagicMock(side_effect=ImportError("none")),
        )
        monkeypatch.setattr(deps, "_find_cosmos_predict2", lambda: None)
        monkeypatch.setattr(deps, "_try_clone_cosmos_predict2", lambda: None)
        with pytest.raises(ImportError, match="cosmos_predict2 not found"):
            deps.ensure_cosmos_predict2()

    def test_success_path_adds_to_sys_path(self, monkeypatch, tmp_path):
        deps._setup_done = False
        sys.modules.pop("cosmos_predict2", None)

        attempts = {"count": 0}

        def fake_import(name):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ImportError("first")
            return types.ModuleType(name)

        monkeypatch.setattr("importlib.import_module", fake_import)
        monkeypatch.setattr(deps, "_find_cosmos_predict2", lambda: str(tmp_path / "repo"))
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
        deps.ensure_cosmos_predict2()
        assert str(tmp_path / "repo") in sys.path

    def test_injects_cosmos_cuda_when_missing(self, monkeypatch, tmp_path):
        deps._setup_done = False
        sys.modules.pop("cosmos_predict2", None)
        sys.modules.pop("cosmos_cuda", None)
        attempts = {"count": 0}

        def fake_import(name):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ImportError("first")
            return types.ModuleType(name)

        monkeypatch.setattr("importlib.import_module", fake_import)
        monkeypatch.setattr(deps, "_find_cosmos_predict2", lambda: str(tmp_path / "repo"))
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        deps.ensure_cosmos_predict2()
        assert "cosmos_cuda" in sys.modules
