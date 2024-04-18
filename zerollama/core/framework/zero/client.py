
import zmq
import traceback
from uuid import uuid4
from queue import Queue

RCVTIMEO = 100000


class Socket(object):
    def __init__(self, context, addr):
        self.addr = addr
        self.socket = context.socket(zmq.REQ)
        self.socket.set_string(zmq.IDENTITY, uuid4().hex)
        self.socket.connect(addr)
        self.socket.RCVTIMEO = RCVTIMEO

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
        self.context = zmq.Context()

    def get(self, addr):
        if addr not in self.queue:
            self.queue[addr] = Queue()

        queue = self.queue[addr]
        if queue.empty():
            return Socket(self.context, addr)
        else:
            return queue.get()

    def put(self, socket, addr):
        if addr not in self.queue:
            self.queue[addr] = Queue()
        queue = self.queue[addr]

        socket.set_timeout(RCVTIMEO)
        queue.put(socket)

    def delete(self, socket):
        socket.close()


socket_pool = SocketPool()


class Timeout(Exception):
    pass


class Client(object):
    def _query(self, addr, data, timeout=None):
        for i in range(3):
            socket = socket_pool.get(addr)
            if timeout is not None:
                socket.set_timeout(timeout)

            try:
                socket.send(data)
                msg = socket.recv()
                socket_pool.put(socket, addr)
                return msg
            except zmq.error.Again:
                socket_pool.delete(socket)

        raise Timeout(f"{addr} timeout")

    def _stream_query(self, addr, data, timeout=None):
        for i in range(3):
            socket = socket_pool.get(addr)
            if timeout is not None:
                socket.set_timeout(timeout)

            try:
                socket.send(data)

                yield socket.recv()
                while socket.getsockopt(zmq.RCVMORE):
                    yield socket.recv()

                socket_pool.put(socket, addr)
                return
            except zmq.error.Again:
                socket_pool.delete(socket)

        raise Timeout(f"{addr} timeout")