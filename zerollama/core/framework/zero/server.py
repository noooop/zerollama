import time
import zmq
import json
import traceback
from multiprocessing import Event, Process, Value
from zerollama.core.framework.zero.client import Timeout


class ZeroServer(object):
    POLL_INTERVAL = 1000

    def __init__(self, port=None, event=None, do_register=True, share_port=None, nameserver_port=None):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        if port is None or port == "random":
            port = socket.bind_to_random_port("tcp://*", min_port=50000, max_port=60000)
        else:
            try:
                socket.bind(f"tcp://*:{port}")
            except zmq.error.ZMQError as e:
                self.port = port
                self.address_in_use = True
                return

        self.port = port
        self.address_in_use = False
        if share_port is not None:
            share_port.value = port

        self.nameserver_port = nameserver_port
        self.context = context
        self.socket = socket
        self.event = event if event is not None else Event()
        self.event.set()
        self.do_register = do_register
        self.share_port = share_port

    def _register(self):
        from zerollama.core.framework.nameserver.client import NameServerClient
        client = NameServerClient(self.nameserver_port)

        server = {"host": "localhost", "port": self.port, "name": self.name, "protocol": self.protocol}
        client.register(server)

    def _deregister(self):
        from zerollama.core.framework.nameserver.client import NameServerClient
        client = NameServerClient(self.nameserver_port)

        server = {"host": "localhost", "port": self.port, "name": self.name, "protocol": self.protocol}
        client.deregister(server)

    def init(self):
        print(f"{self.__class__.__name__} is running!", "port:", self.port)

    def clean_up(self):
        print(f"{self.__class__.__name__} clean_up!")

    def process(self):
        msg = self.socket.recv()
        print(msg)
        self.socket.send(b"ok")

    def run(self):
        self.init()
        if self.address_in_use:
            print(f"Address in use (addr='tcp://*:{self.port}')")
            return

        if self.do_register:
            self._register()

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        while self.event.is_set():
            try:
                socks = dict(poller.poll(self.POLL_INTERVAL))
            except (KeyboardInterrupt, EOFError):
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


class Z_MethodZeroServer(ZeroServer):
    def process(self):
        msg = self.socket.recv()
        try:
            msg = json.loads(msg)
        except json.JSONDecodeError as e:
            traceback.print_exc()
            self.handle_error(err_msg=str(e))
            return

        if "method" not in msg:
            self.handle_error(err_msg="'method' not in msg")
            return

        method_name = msg["method"]
        method = getattr(self, "z_"+method_name, self.default)

        try:
            method(msg)
        except Exception as e:
            self.handle_error(err_msg=str(e))
            return

    def default(self, msg):
        method_name = msg["method"]
        self.handle_error(err_msg=f"method [{method_name}] not supported.")

    def z_support_methods(self, msg):
        support_methods = [m[2:] for m in dir(self) if m.startswith('z_')]

        response = json.dumps({
            "state": "ok",
            "support_methods": support_methods
        }).encode('utf8')

        self.socket.send(response)


class ZeroServerProcess(Process):
    def __init__(self, server_class, server_kwargs=None, event=None, ignore_warnings=False):
        Process.__init__(self)
        if event is None:
            self.event = Event()
        else:
            self.event = event

        self.server_class = server_class
        self.server_kwargs = server_kwargs or dict()
        self.server = None
        self.share_port = Value('i', -1)
        self.ignore_warnings = ignore_warnings

    def wait_port_available(self, timeout=10000):
        t = timeout + time.time()

        while time.time() < t:
            time.sleep(0.1)
            if self.share_port.value != -1:
                return self.share_port.value

    def run(self):
        if self.ignore_warnings:
            import warnings
            warnings.filterwarnings("ignore")

        self.server_kwargs["event"] = self.event
        self.server_kwargs["share_port"] = self.share_port

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

    def wait(self):
        try:
            self.join()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self.terminate()


if __name__ == '__main__':
    server_class = "zerollama.core.framework.zero.server:ZeroServer"

    h1 = ZeroServerProcess(server_class, {"do_register": False})
    h2 = ZeroServerProcess(server_class, {"do_register": False})
    h3 = ZeroServerProcess(server_class, {"do_register": False})

    h1.start()
    h2.start()
    h3.start()

    time.sleep(10)

    h1.terminate()
    h2.terminate()
    h3.terminate()

    server_class = "zerollama.core.framework.zero.server:Z_MethodZeroServer"
    h = ZeroServerProcess(server_class, {"do_register": False, "port": 9527})
    h.start()
    h.wait()

