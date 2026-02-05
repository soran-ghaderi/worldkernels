"""WorldKernels CLI."""

from __future__ import annotations

import sys


def app() -> None:
    """Main CLI entry point."""
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print(
            """worldkernels - GPU-first world model simulation engine

Usage:
    worldkernels serve [--host HOST] [--port PORT]   Start HTTP/WebSocket server
    worldkernels version                             Show version
    worldkernels --help                              Show this help

Examples:
    worldkernels serve --host 0.0.0.0 --port 8000
"""
        )
        return

    if args[0] in ("version", "--version", "-V"):
        from worldkernels import __version__

        print(f"worldkernels {__version__}")
        return

    if args[0] == "serve":
        print("Server not yet implemented. Coming soon!")
        print("Install with: pip install worldkernels[serve]")
        return

    print(f"Unknown command: {args[0]}")
    print("Run 'worldkernels --help' for usage.")
    sys.exit(1)


if __name__ == "__main__":
    app()
