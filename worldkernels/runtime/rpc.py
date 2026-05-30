r"""Length-prefixed msgpack RPC over a Unix domain socket.

Wire format: 4-byte big-endian length header + msgpack body. Each message is
a dict with ``id``, ``op``, ``args`` (request) or ``id``, ``ok``, ``result|error``
(response). Synchronous request/response — the worker handles one call at a
time per session, matching the engine's per-session step contract.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack

log = logging.getLogger(__name__)

__all__ = ["RpcServer", "RpcClient", "RpcError", "default_socket_path"]

_HEADER = struct.Struct(">I")
_MAX_FRAME = 1 << 26


class RpcError(RuntimeError):
    r"""Raised on the client side when the worker returned an error envelope."""

    def __init__(self, op: str, cls: str, message: str) -> None:
        super().__init__(f"{op}: {cls}: {message}")
        self.op = op
        self.exc_cls = cls
        self.message = message


def default_socket_path(model_id: str) -> Path:
    slug = model_id.replace("/", "_").replace(":", "_")
    suffix = uuid.uuid4().hex[:8]
    return Path(tempfile.gettempdir()) / f"wk-{slug}-{suffix}.sock"


@dataclass
class _Frame:
    payload: dict[str, Any]


def _send(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = msgpack.packb(payload, use_bin_type=True)
    if len(body) > _MAX_FRAME:
        raise RpcError("send", "RpcError", f"frame too large: {len(body)} bytes")
    sock.sendall(_HEADER.pack(len(body)) + body)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed during recv")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exactly(sock, _HEADER.size)
    (length,) = _HEADER.unpack(header)
    if length > _MAX_FRAME:
        raise RpcError("recv", "RpcError", f"frame too large: {length} bytes")
    body = _recv_exactly(sock, length)
    return msgpack.unpackb(body, raw=False)


class RpcClient:
    r"""Synchronous client. One in-flight call at a time; safe to reuse across calls."""

    def __init__(self, socket_path: Path, connect_timeout: float = 30.0) -> None:
        self.socket_path = Path(socket_path)
        self._sock: socket.socket | None = None
        self._connect_timeout = connect_timeout

    def connect(self) -> None:
        deadline = self._connect_timeout
        import time as _time

        end = _time.monotonic() + deadline
        last_exc: Exception | None = None
        while _time.monotonic() < end:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(str(self.socket_path))
                self._sock = s
                return
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                last_exc = exc
                _time.sleep(0.05)
        raise ConnectionError(f"could not connect to {self.socket_path}: {last_exc}")

    def call(self, op: str, **args: Any) -> Any:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        req = {"id": uuid.uuid4().hex, "op": op, "args": args}
        _send(self._sock, req)
        resp = _recv(self._sock)
        if resp.get("ok"):
            return resp.get("result")
        err = resp.get("error", {})
        raise RpcError(op, err.get("cls", "RpcError"), err.get("message", "unknown"))

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None


class RpcServer:
    r"""Listens on a Unix socket, dispatches ops to a handler callable.

    Handler signature: ``handler(op: str, args: dict) -> Any``. The server
    handles one client at a time; if the engine reconnects, the next ``accept``
    returns. Designed for the engine↔worker 1:1 pairing.
    """

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = Path(socket_path)
        self._server: socket.socket | None = None
        self._shutdown = False

    def serve(self, handler) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        self._server.listen(1)
        log.info("rpc.server listening on %s", self.socket_path)

        while not self._shutdown:
            try:
                client, _ = self._server.accept()
            except OSError:
                break
            self._serve_client(client, handler)

    def _serve_client(self, client: socket.socket, handler) -> None:
        try:
            while not self._shutdown:
                try:
                    req = _recv(client)
                except (ConnectionError, EOFError):
                    return
                op = req.get("op", "")
                args = req.get("args", {})
                req_id = req.get("id")
                resp: dict[str, Any]
                try:
                    result = handler(op, args)
                    resp = {"id": req_id, "ok": True, "result": result}
                except Exception as exc:
                    resp = {
                        "id": req_id,
                        "ok": False,
                        "error": {"cls": type(exc).__name__, "message": str(exc)},
                    }
                _send(client, resp)
                if op == "close":
                    self._shutdown = True
                    return
        finally:
            client.close()

    def stop(self) -> None:
        self._shutdown = True
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        try:
            self.socket_path.unlink(missing_ok=True)
        except OSError:
            pass
