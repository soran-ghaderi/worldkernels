r"""CLI cold-start budget: heavy libs must not load on parse/help paths."""

from __future__ import annotations

import subprocess
import sys

_HEAVY = ("torch", "rich", "numpy", "transformers", "diffusers")


class TestImportBudget:
    def test_build_parser_imports_no_heavy_libs(self):
        code = (
            "import sys\n"
            "from worldkernels.cli.main import build_parser\n"
            "build_parser()\n"
            f"heavy = [m for m in {_HEAVY!r} if m in sys.modules]\n"
            "assert not heavy, f'heavy imports on parser build: {heavy}'\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=30)

    def test_help_exits_zero_without_torch(self):
        out = subprocess.run(
            [sys.executable, "-X", "importtime", "-m", "worldkernels.cli.main", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert out.returncode == 0
        assert "import time:" in out.stderr
        assert "torch" not in out.stderr
