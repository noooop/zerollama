
import zmq
from uuid import uuid4
from queue import Queue


class Socket(object):
    def __init__(self, context, addr):
        self.addr = addr
        self.socket = context.socket(zmq.REQ)
        self.socket.set_string(zmq.IDENTITY, uuid4().hex)
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
        queue.put(socket)

    def delete(self, socket):
        try:
            print("delete_socket", socket.addr)
            socket.close()
        except Exception:
            pass


socket_pool = SocketPool()


class Client(object):
    def _query(self, addr, data):
        socket = socket_pool.get(addr)

        try:
            socket.send(data)
            msg = socket.recv()
            socket_pool.put(socket, addr)
            return msg
        except Exception:
            socket_pool.delete(socket)

    def _stream_query(self, addr, data):
        socket = socket_pool.get(addr)

        try:
            socket.send(data)

            yield socket.recv()
            while socket.getsockopt(zmq.RCVMORE):
                yield socket.recv()

            socket_pool.put(socket, addr)
        except Exception:
            socket_pool.delete(socket)