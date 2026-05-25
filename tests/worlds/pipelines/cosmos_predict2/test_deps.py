r"""Tests for worldkernels/worlds/pipelines/cosmos_predict2/deps.py."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from worldkernels.worlds.pipelines.cosmos_predict2 import deps


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    snapshot = dict(sys.modules)
    setup = deps._setup_done
    yield
    deps._setup_done = setup
    for key in list(sys.modules):
        if key not in snapshot:
            del sys.modules[key]


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
                str(tmp_path / "empty_env")  # noqa: E501
                if key == "COSMOS_PREDICT2_PATH"
                else default
            ),
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
        deps._setup_done = False  # noqa: E501
        sys.modules["cosmos_predict2"] = types.ModuleType("cosmos_predict2")
        monkeypatch.setattr(
            deps, "_find_cosmos_predict2", lambda: pytest.fail("must not be called")
        )
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
