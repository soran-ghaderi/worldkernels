r"""Bootstrap-specific errors. Surfaced cleanly to CLI/HTTP without tracebacks."""

from __future__ import annotations

from worldkernels.core.errors import WorldKernelError


class BootstrapError(WorldKernelError):
    pass


class FetchDisabledError(BootstrapError):
    def __init__(self, what: str, hint: str) -> None:
        super().__init__(f"{what} (auto-fetch disabled). {hint}")


class AuthRequiredError(BootstrapError):
    def __init__(self, repo: str) -> None:
        super().__init__(
            f"Model '{repo}' is gated. Run `huggingface-cli login` and retry."
        )


class ModelNotFoundError(BootstrapError):
    pass
