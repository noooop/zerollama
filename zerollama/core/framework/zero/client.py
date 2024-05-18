
import json
import gevent
import shortuuid
import numpy as np
from os import getpid
import zmq.green as zmq
from queue import Queue
from zerollama.core.framework.zero.protocol import ZeroMSQ
from zerollama.core.framework.zero.protocol import ZeroServerResponse, Timeout


class Socket(object):
    def __init__(self, context, addr):
        self.addr = addr
        self.socket = context.socket(zmq.DEALER)
        self.socket.connect(addr)

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

    def _query(self, req, req_payload, n_try=3, timeout=None):
        _timeout = timeout or getattr(self, "timeout", None)

        for i in range(n_try):
            req_id = f"{shortuuid.random(length=22)}".encode("utf-8")
            socket = socket_pool.get(self.addr)

            try:
                with gevent.Timeout(_timeout):
                    socket.send_multipart([req_id, req]+req_payload)
                    msg = socket.recv_multipart()
                    socket_pool.put(socket, self.addr)
                    return msg
            except gevent.timeout.Timeout:
                socket_pool.delete(socket)

        raise Timeout(f"{self.addr} timeout")

    def _stream_query(self, req, req_payload, n_try=3, timeout=None):
        _timeout = timeout or getattr(self, "timeout", None)

        for i in range(n_try):
            req_id = f"{shortuuid.random(length=22)}".encode("utf-8")
            socket = socket_pool.get(self.addr)
            try:
                with gevent.Timeout(_timeout):
                    socket.send_multipart([req_id, req]+req_payload)
                    out = socket.recv_multipart()
            except gevent.timeout.Timeout:
                socket_pool.delete(socket)
                continue

            rep_id, msg, *payload = out
            req_id, rcv_more, rep_id = rep_id[:22], rep_id[22:23], rep_id[23:]
            yield out

            while rcv_more == b"M":
                try:
                    with gevent.Timeout(_timeout):
                        out = socket.recv_multipart()
                        rep_id, msg, *payload = out
                        req_id, rcv_more, rep_id = rep_id[:22], rep_id[22:23], rep_id[23:]
                        yield out
                except gevent.timeout.Timeout:
                    socket_pool.delete(socket)
                    raise Timeout(f"{self.addr} timeout")
            else:
                return

        raise Timeout(f"{self.addr} timeout")

    def stream_query(self, data, **kwargs):
        req, req_payload = ZeroMSQ.load(data)
        for req_id, msg, *payload in self._stream_query(req, req_payload, **kwargs):
            yield ZeroServerResponse(**ZeroMSQ.unload(msg, payload))

    def query(self, data, **kwargs):
        req, req_payload = ZeroMSQ.load(data)
        req_id, msg, *payload = self._query(req, req_payload, **kwargs)
        return ZeroServerResponse(**ZeroMSQ.unload(msg, payload))


class Z_Client(Client):
    def support_methods(self, **kwargs):
        data = {"method": "support_methods"}
        return self.query(data, **kwargs)


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

