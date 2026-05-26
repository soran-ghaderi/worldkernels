r"""Per-model lockfile + deps hash."""

from __future__ import annotations

from worldkernels.runtime import locks


def test_deps_hash_stable_across_order(monkeypatch):
    a = locks.deps_hash(["transformers==4.46", "diffusers==0.30"], "abi", "cuda")
    b = locks.deps_hash(["diffusers==0.30", "transformers==4.46"], "abi", "cuda")
    assert a == b


def test_deps_hash_diverges_on_abi_change():
    a = locks.deps_hash(["transformers==4.46"], "abi-A", "cuda")
    b = locks.deps_hash(["transformers==4.46"], "abi-B", "cuda")
    assert a != b


def test_envlock_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLDKERNELS_HOME", str(tmp_path))
    lock = locks.EnvLock(
        model_id="org/m",
        deps_hash="abc",
        torch_abi="2.4-cu121",
        device="cuda",
        requirements=["transformers==4.46"],
    )
    lock.write()
    got = locks.EnvLock.read("org/m")
    assert got is not None
    assert got.deps_hash == "abc"
    assert got.requirements == ["transformers==4.46"]
