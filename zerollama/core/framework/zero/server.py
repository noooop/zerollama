import zmq
import json
from multiprocessing import Event, Process
from .client import Timeout


class ZeroServer(object):
    POLL_INTERVAL = 1000

    def __init__(self, port=None, event=None, do_register=True):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        if port is None:
            port = socket.bind_to_random_port("tcp://*", min_port=50000, max_port=60000)
        else:
            socket.bind(f"tcp://*:{port}")

        self.port = port
        self.context = context
        self.socket = socket
        self.event = event if event is not None else Event()
        self.event.set()
        self.do_register = do_register

    def _register(self):
        from zerollama.core.framework.nameserver.client import NameServerClient
        client = NameServerClient()

        server = {"host": "localhost", "port": self.port, "name": self.name, "protocol": self.protocol}
        client.register(server)

    def _deregister(self):
        from zerollama.core.framework.nameserver.client import NameServerClient
        client = NameServerClient()

        server = {"host": "localhost", "port": self.port, "name": self.name, "protocol": self.protocol}
        client.deregister(server)

    def init(self):
        print(f"{self.__class__.__name__} is running!")

    def clean_up(self):
        print(f"{self.__class__.__name__} clean_up!")

    def process(self):
        msg = self.socket.recv()
        print(msg)
        self.socket.send(b"ok")

    def run(self):
        self.init()

        if self.do_register:
            self._register()

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        while self.event.is_set():
            try:
                socks = dict(poller.poll(self.POLL_INTERVAL))
            except KeyboardInterrupt:
                break

            if socks.get(self.socket) == zmq.POLLIN:
                self.process()

        if self.do_register:
            try:
                self._deregister()
            except Timeout:
                pass

        self.clean_up()

    def handle_error(self, err_msg):
        response = json.dumps({
            "state": "error",
            "msg": err_msg
        }).encode('utf8')

        self.socket.send(response)


class ZeroServerProcess(Process):
    def __init__(self, server_class, server_kwargs=None, event=None):
        Process.__init__(self)
        if event is None:
            self.event = Event()
        else:
            self.event = event

        self.server_class = server_class
        self.server_kwargs = server_kwargs or dict()
        self.server = None

    def run(self):
        self.server_kwargs["event"] = self.event

        if isinstance(self.server_class, str):
            module_name, class_name = self.server_class.split(":")
            import importlib
            module = importlib.import_module(module_name)
            self.server_class = getattr(module, class_name)

        self.server = self.server_class(**self.server_kwargs)
        self.server.run()

    def terminate(self):
        self.event.clear()
        self.join()


if __name__ == '__main__':
    import time

    server_class = "zerollama.core.framework.zero.server:ZeroServer"

    h1 = ZeroServerProcess(server_class, {"do_register": False}, Event())
    h2 = ZeroServerProcess(server_class, {"do_register": False}, Event())
    h3 = ZeroServerProcess(server_class, {"do_register": False}, Event())

    h1.start()
    h2.start()
    h3.start()

    time.sleep(10)

    h1.terminate()
    h2.terminate()
    h3.terminate()