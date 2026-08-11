r"""Back-compat shim: handlers moved to worldkernels.cli.commands.bench."""

from worldkernels.cli.commands.bench import (  # noqa: F401
    bench_env,
    run_latency,
    run_profile,
    run_startup,
    run_throughput,
    run_vram,
)
