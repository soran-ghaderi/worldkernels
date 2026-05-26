r"""Round-trip an RPC request/response over a Unix socket."""

from __future__ import annotations

import sys
import threading
import time

import pytest

from worldkernels.runtime.rpc import RpcClient, RpcError, RpcServer, default_socket_path


@pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets only")
class TestRpcRoundTrip:
    def test_simple_call(self, tmp_path):
        sock = tmp_path / "rpc.sock"
        server = RpcServer(sock)

        def handler(op, args):
            if op == "echo":
                return {"got": args.get("msg")}
            if op == "close":
                return {"ok": True}
            raise ValueError(f"unknown op {op}")

        t = threading.Thread(target=server.serve, args=(handler,), daemon=True)
        t.start()

        client = RpcClient(sock)
        client.connect()
        out = client.call("echo", msg="hello")
        assert out == {"got": "hello"}
        client.call("close")
        t.join(timeout=2.0)
        client.close()

    def test_error_propagation(self, tmp_path):
        sock = tmp_path / "rpc.sock"
        server = RpcServer(sock)

        def handler(op, args):
            if op == "boom":
                raise ValueError("nope")
            if op == "close":
                return {"ok": True}
            return None

        t = threading.Thread(target=server.serve, args=(handler,), daemon=True)
        t.start()

        client = RpcClient(sock)
        client.connect()
        with pytest.raises(RpcError) as ei:
            client.call("boom")
        assert "nope" in str(ei.value)
        assert ei.value.exc_cls == "ValueError"
        client.call("close")
        client.close()
        t.join(timeout=2.0)


def test_default_socket_path_is_unique():
    a = default_socket_path("foo")
    time.sleep(0.001)
    b = default_socket_path("foo")
    assert a != b
