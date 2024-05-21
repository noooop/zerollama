
import zmq.green as zmq
import json
import shortuuid
import numpy as np
from os import getpid
from queue import Queue
from zerollama.core.framework.zero.protocol import ZeroServerResponse, ZeroServerResponseOkWithPayload, Timeout


class Socket(object):
    def __init__(self, context, addr):
        self.addr = addr
        self.socket = context.socket(zmq.DEALER)
        self.socket.connect(addr)

    def set_timeout(self, timeout):
        self.socket.RCVTIMEO = timeout

    def send(self, data):
        self.socket.send(data, copy=False)

    def send_multipart(self, data):
        self.socket.send_multipart(data, copy=False)

    def recv(self):
        return self.socket.recv()

    def recv_multipart(self):
        return self.socket.recv_multipart()

    def close(self):
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()

    def getsockopt(self, opt):
        return self.socket.getsockopt(opt)


class SocketPool(object):
    def __init__(self):
        self.queue = {}
        self.context = zmq.Context.instance()
        self._pid = getpid()

    def reinit(self):
        self.queue = {}
        self.context = zmq.Context.instance()
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

    def _query(self, data, timeout=None):
        for i in range(3):
            socket = socket_pool.get(self.addr)

            if getattr(self, "timeout", None) is not None:
                socket.set_timeout(self.timeout)

            if timeout is not None:
                socket.set_timeout(timeout)

            try:
                req_id = f"{shortuuid.random(length=22)}".encode("utf-8")
                socket.send_multipart([req_id, data])
                msg = socket.recv_multipart()
                socket_pool.put(socket, self.addr)
                return msg
            except zmq.error.Again:
                socket_pool.delete(socket)

        raise Timeout(f"{self.addr} timeout")

    def _stream_query(self, data, timeout=None):
        for i in range(3):
            socket = socket_pool.get(self.addr)

            if getattr(self, "timeout", None) is not None:
                socket.set_timeout(self.timeout)

            if timeout is not None:
                socket.set_timeout(timeout)

            try:
                req_id = f"{shortuuid.random(length=22)}".encode("utf-8")
                socket.send_multipart([req_id, data])
                out = socket.recv_multipart()
                rep_id, msg, *payload = out
                req_id, rcv_more, rep_id = rep_id[:22], rep_id[22:23], rep_id[23:]

                yield out

                while rcv_more == b"M":
                    out = socket.recv_multipart()
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

    def stream_query(self, data, **kwargs):
        data = json.dumps(data).encode('utf8')
        for req_id, msg, *payload in self._stream_query(data=data, **kwargs):
            yield self._load(req_id, msg, payload)

    def query(self, data, **kwargs):
        data = json.dumps(data).encode('utf8')

        req_id, msg, *payload = self._query(data=data, **kwargs)
        return self._load(req_id, msg, payload)


class Z_Client(Client):
    def support_methods(self):
        data = {"method": "support_methods"}
        return self.query(data)


if __name__ == '__main__':
    from pprint import pprint
    client = Z_Client("tcp://localhost:9527")

    print("="*80)
    pprint(client.query({"no_method": ""}).dict())
    print("=" * 80)
    pprint(client.support_methods().dict())
    print("=" * 80)
    data = {"method": "method_not_supported"}
    pprint(client.query(data).dict())

