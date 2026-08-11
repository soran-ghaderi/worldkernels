r"""Back-compat shim: handler moved to worldkernels.cli.commands.run."""

from worldkernels.cli.commands.run import (  # noqa: F401
    _raw_to_arrays,
    _save_frames,
    _save_video,
    run_session,
)
