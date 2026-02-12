from __future__ import annotations

import json

import zmq

from quantbt.messaging.protocol import Message


class ZmqClient:
    """DEALER socket client for communicating with tradecore ROUTER."""

    def __init__(self, address: str = "tcp://127.0.0.1:5555",
                 identity: str = "quantbt-1") -> None:
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.DEALER)
        self._socket.setsockopt_string(zmq.IDENTITY, identity)
        self._socket.connect(address)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    def send(self, message: Message) -> None:
        payload = json.dumps(message.to_dict()).encode("utf-8")
        self._socket.send_multipart([b"", payload])

    def recv(self, timeout_ms: int = 100) -> dict | None:
        events = dict(self._poller.poll(timeout_ms))
        if self._socket in events:
            frames = self._socket.recv_multipart()
            # DEALER receives [empty delimiter, data]
            data = frames[-1]
            return json.loads(data.decode("utf-8"))
        return None

    def close(self) -> None:
        self._socket.close()
