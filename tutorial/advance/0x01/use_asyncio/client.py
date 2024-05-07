
import zmq
import platform
import asyncio
from zmq.asyncio import Context

import json
import shortuuid
import numpy as np
from os import getpid
from queue import Queue
from zerollama.core.framework.zero.protocol import ZeroServerResponse, ZeroServerResponseOkWithPayload, Timeout

plat = platform.system().lower()
if plat == 'windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class Socket(object):
    def __init__(self, context, addr):
        self.addr = addr
        self.socket = context.socket(zmq.DEALER)
        self.socket.connect(addr)

    def set_timeout(self, timeout):
        self.socket.RCVTIMEO = timeout

    async def send(self, data):
        await self.socket.send(data, copy=False)

    async def send_multipart(self, data):
        await self.socket.send_multipart(data, copy=False)

    async def recv(self):
        return await self.socket.recv()

    async def recv_multipart(self):
        return await self.socket.recv_multipart()

    def close(self):
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()

    def getsockopt(self, opt):
        return self.socket.getsockopt(opt)


class SocketPool(object):
    def __init__(self):
        self.queue = {}
        self.context = Context.instance()
        self._pid = getpid()

    def reinit(self):
        self.queue = {}
        self.context = Context.instance()
        self._pid = getpid()

    def get(self, addr):
        if self._pid != getpid():
            self.reinit()

        if addr not in self.queue:
            self.queue[addr] = Queue()

        queue = self.queue[addr]

        if queue.empty():
            return Socket(self.context, addr)
        else:
            return queue.get()

    def put(self, socket, addr):
        if self._pid != getpid():
            self.reinit()
        else:
            if addr not in self.queue:
                self.queue[addr] = Queue()
            queue = self.queue[addr]
            queue.put(socket)

    def delete(self, socket):
        if self._pid != getpid():
            self.reinit()
        else:
            socket.close()


socket_pool = SocketPool()


class Client(object):
    timeout = 100000

    def __init__(self, addr):
        self.addr = addr

    async def _query(self, data, timeout=None):
        for i in range(3):
            socket = socket_pool.get(self.addr)

            if getattr(self, "timeout", None) is not None:
                socket.set_timeout(self.timeout)

            if timeout is not None:
                socket.set_timeout(timeout)

            try:
                req_id = f"{shortuuid.random(length=22)}".encode("utf-8")
                await socket.send_multipart([req_id, data])
                msg = await socket.recv_multipart()
                socket_pool.put(socket, self.addr)
                return msg
            except zmq.error.Again:
                socket_pool.delete(socket)

        raise Timeout(f"{self.addr} timeout")

    async def _stream_query(self, data, timeout=None):
        for i in range(3):
            socket = socket_pool.get(self.addr)

            if getattr(self, "timeout", None) is not None:
                socket.set_timeout(self.timeout)

            if timeout is not None:
                socket.set_timeout(timeout)

            try:
                req_id = f"{shortuuid.random(length=22)}".encode("utf-8")
                await socket.send_multipart([req_id, data])
                out = await socket.recv_multipart()
                rep_id, msg, *payload = out
                req_id, rcv_more, rep_id = rep_id[:22], rep_id[22:23], rep_id[23:]

                yield out

                while rcv_more == b"M":
                    out = await socket.recv_multipart()
                    rep_id, msg, *payload = out
                    req_id, rcv_more, rep_id = rep_id[:22], rep_id[22:23], rep_id[23:]
                    yield out

                socket_pool.put(socket, self.addr)
            except zmq.error.Again:
                socket_pool.delete(socket)
            else:
                return

        raise Timeout(f"{self.addr} timeout")

    def _load(self, req_id, msg, payload):
        msg = json.loads(msg)
        if len(payload) > 0:
            msg = ZeroServerResponseOkWithPayload(**msg)
            msg.payload = []
            for m, p in zip(msg.meta, payload):
                msg.payload.append(np.frombuffer(p, dtype=m.dtype).reshape(m.shape))
            return msg
        else:
            msg = ZeroServerResponse(**msg)
            return msg

    async def stream_query(self, data, **kwargs):
        data = json.dumps(data).encode('utf8')
        async for req_id, msg, *payload in self._stream_query(data=data, **kwargs):
            yield self._load(req_id, msg, payload)

    async def query(self, data, **kwargs):
        data = json.dumps(data).encode('utf8')

        req_id, msg, *payload = await self._query(data=data, **kwargs)
        return self._load(req_id, msg, payload)


class Z_Client(Client):
    async def support_methods(self):
        data = {"method": "support_methods"}
        return self.query(data)

