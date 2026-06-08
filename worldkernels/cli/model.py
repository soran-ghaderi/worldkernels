r"""Back-compat shim: handlers moved to worldkernels.cli.commands.model."""

from worldkernels.cli.commands.model import (  # noqa: F401
    run_download,
    run_export,
    run_inspect,
    run_list,
    run_remove,
)
