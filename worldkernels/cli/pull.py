r"""Back-compat shim: handlers moved to worldkernels.cli.commands.pull."""

from worldkernels.cli.commands.pull import (  # noqa: F401
    _human,
    run_models,
    run_pull,
    run_rm,
)
