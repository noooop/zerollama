
import zmq
import json
from uuid import uuid4
from queue import Queue


class Socket(object):
    def __init__(self, context, addr):
        self.addr = addr
        self.uuid = uuid4().hex.encode("utf-8")
        self.socket = context.socket(zmq.DEALER)
        self.socket.set_string(zmq.IDENTITY, uuid4().hex)
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
        socket.close()


socket_pool = SocketPool()


class Timeout(Exception):
    pass


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
                socket.send(data)
                msg = socket.recv()
                socket_pool.put(socket, self.addr)
                return msg
            except zmq.error.Again:
                socket_pool.delete(socket)

        raise Timeout(f"{self.addr} timeout")

    def _stream_query(self, data, timeout=None):
        for i in range(3):
            socket = socket_pool.get(self.addr)
            if timeout is not None:
                socket.set_timeout(timeout)

            try:
                socket.send(data)

                data = socket.recv()
                data = json.loads(data)
                yield data

                while not data["done"]:
                    data = socket.recv()
                    data = json.loads(data)
                    yield data

                    if data["done"]:
                        break

                socket_pool.put(socket, self.addr)
                return
            except zmq.error.Again:
                socket_pool.delete(socket)

        raise Timeout(f"{self.addr} timeout")

    def json_query(self, data, debug=False, **kwargs):
        data = json.dumps(data).encode('utf8')

        if debug:
            print(data)

        msg = self._query(data=data, **kwargs)
        msg = json.loads(msg)
        return msg

    def json_stream_query(self, data, debug=False, **kwargs):
        data = json.dumps(data).encode('utf8')

        if debug:
            print(data)

        for part in self._stream_query(data=data, **kwargs):
            yield part


class Z_Client(Client):
    def support_methods(self):
        data = {"method": "support_methods"}
        return self.json_query(data)


if __name__ == '__main__':
    from pprint import pprint
    client = Z_Client("tcp://localhost:9527")

    pprint(client.support_methods())

    data = {"method": "method_not_supported"}
    pprint(client.json_query(data))

