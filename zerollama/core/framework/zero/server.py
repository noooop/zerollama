
import json
import time
import gevent
import signal
import greenlet
import traceback
import zmq.green as zmq
from zmq.error import ZMQError
from gevent.pool import Pool
from multiprocessing import Event, Process, Value, Pipe, Manager

from zerollama.core.framework.zero.protocol import (
    Timeout,
    ZeroMSQ,
    convert_errors,
    ValidationError,
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)


class ZeroServer(object):
    POLL_INTERVAL = 1000
    POOL_SIZE = 64

    def __init__(self, name=None, protocol=None, port=None, event=None, do_register=True, share_port=None,
                 nameserver_port=None):
        context = zmq.Context.instance()
        socket = context.socket(zmq.ROUTER)
        if port is None or port == "random":
            port = socket.bind_to_random_port("tcp://*", min_port=50000, max_port=60000)
        else:
            try:
                socket.bind(f"tcp://*:{port}")
            except ZMQError as e:
                self.port = port
                self.address_in_use = True
                share_port.value = -2
                return

        self.port = port
        self.address_in_use = False
        if share_port is not None:
            share_port.value = port

        self.name = name
        self.protocol = protocol or getattr(self, "protocol", None)
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

    def process(self, msg):
        try:
            uuid, req_id, msg, *payload = msg
        except Exception:
            traceback.print_exc()
            return

        return self.socket.send_multipart([uuid, req_id, b"ok"])

    def start(self):
        self.init()

        gevent.signal_handler(signal.SIGTERM, self.close)
        gevent.signal_handler(signal.SIGINT, self.close)

        self.run()

    def run(self):
        if self.address_in_use:
            raise RuntimeError(f"Address in use (addr='tcp://*:{self.port}')")

        if self.do_register:
            self._register()

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        def get_task():
            while self.event.is_set():
                try:
                    socks = dict(poller.poll(self.POLL_INTERVAL))
                except (KeyboardInterrupt, EOFError):
                    return

                if socks.get(self.socket) == zmq.POLLIN:
                    msg = self.socket.recv_multipart()
                    yield msg

        p = Pool(self.POOL_SIZE)

        for x in p.imap_unordered(self.process, get_task()):
            pass

        if self.do_register:
            try:
                self._deregister()
            except Timeout:
                pass

        self.clean_up()

    def handle_error(self, err_msg, uuid=None, req_id=None, req=None):
        if req is not None:
            uuid, req_id = req.uuid, req.req_id

        if isinstance(err_msg, ValidationError):
            err_msg = convert_errors(err_msg)

        rep = ZeroServerResponseError(msg=err_msg)

        data, payload = ZeroMSQ.load(rep.dict())
        self.socket.send_multipart([uuid, req_id, data] + payload)

    def zero_send(self, req: ZeroServerRequest, rep: ZeroServerResponse):
        if isinstance(rep, ZeroServerStreamResponseOk):
            rep_id = req.req_id + (b"M" if rep.snd_more else b"N") + str(rep.rep_id).encode("utf-8")
        else:
            rep_id = req.req_id

        data, payload = ZeroMSQ.load(rep.dict())

        self.socket.send_multipart([req.uuid, rep_id, data] + payload)

    def close(self):
        self.event.clear()


class Z_MethodZeroServer(ZeroServer):
    def process(self, data):
        try:
            uuid, req_id, data, *payload = data
        except Exception:
            traceback.print_exc()
            return

        try:
            msg = ZeroMSQ.unload(data, payload)
        except json.JSONDecodeError as e:
            self.handle_error(str(e), uuid, req_id)
            return

        try:
            req = ZeroServerRequest(**msg)
        except ValidationError as e:
            self.handle_error(e, uuid, req_id)
            return

        req.uuid = uuid
        req.req_id = req_id

        method_name = msg["method"]
        method = getattr(self, "z_" + method_name, self.default)

        try:
            method(req)
        except ValidationError as e:
            self.handle_error(e, req=req)
        except Exception as e:
            traceback.print_exc()
            self.handle_error(str(e), req=req)

    def default(self, req: ZeroServerRequest):
        method_name = req.method
        err_msg = f"method [{method_name}] not supported."
        self.handle_error(err_msg, req=req)

    def z_support_methods(self, req: ZeroServerRequest):
        support_methods = [m[2:] for m in dir(self) if m.startswith('z_')]
        rep = ZeroServerResponseOk(msg={
            "name": self.__class__.__name__,
            "support_methods": support_methods
        })
        self.zero_send(req, rep)

    def z_info(self, req: ZeroServerRequest):
        rep = ZeroServerResponseOk(msg={})
        self.zero_send(req, rep)


class ZeroServerProcess(Process):
    """
    Class which returns child Exceptions to Parent.
    https://stackoverflow.com/a/33599967/4992248

    status:
    prepare -> started -> initial -> running -> error or stopped
    """

    def __init__(self, server_class, server_kwargs=None, event=None, ignore_warnings=False, debug=False):
        Process.__init__(self)

        if event is None:
            self.event = Event()
        else:
            self.event = event

        self.debug = debug
        self.server_class = server_class
        self.server_kwargs = server_kwargs or dict()
        self.server = None
        self.share_port = Value('i', -1)
        self.ignore_warnings = ignore_warnings

        self._status = Manager().dict()
        self._parent_conn, self._child_conn = Pipe()
        self._exception = None

        self._set_status("prepare")

        # slow log
        self.__last_switch_time_ms, self.__last_switch_cpu_tick = None, None
        self.__threshold_ms = 0.1

    def _slow_log(self, event, args):
        if event not in {'switch', 'throw'}:
            return

        origin, target = args

        then_time, then_tick = self.__last_switch_time_ms, self.__last_switch_cpu_tick
        now_time, now_tick = time.time(), time.process_time()
        self.__last_switch_time_ms, self.__last_switch_cpu_tick = now_time, now_tick

        if None in (then_time, then_tick):
            return

        if origin is gevent.hub.get_hub():
            return

        elapsed_ms = now_time - then_time
        elapsed_ticks = now_tick - then_tick

        if elapsed_ms <= self.__threshold_ms:
            return

        print(f"[SLOW LOG] Transfer from {origin} to {target} with {event} elapsed_ms: [{elapsed_ms}] elapsed_ticks [{elapsed_ticks}]")

    def _set_status(self, status):
        self._status['status'] = status

    def wait_port_available(self, timeout=10000):
        t = timeout + time.time()

        while time.time() < t:
            time.sleep(0.1)
            if self.share_port.value != -1:
                return self.share_port.value

    def run(self):
        self._set_status("started")

        try:
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
        except (FileNotFoundError, EnvironmentError) as e:
            print(e)
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return
        except Exception as e:
            print(e)
            traceback.print_exc()
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return

        self._set_status("initial")

        try:
            self.server.init()
        except (FileNotFoundError, EnvironmentError) as e:
            print(e)
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return
        except Exception as e:
            print(e)
            traceback.print_exc()
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return

        self._set_status("running")

        if self.debug:
            greenlet.settrace(self._slow_log)

        gevent.signal_handler(signal.SIGTERM, self.close)
        gevent.signal_handler(signal.SIGINT, self.close)

        try:
            self.server.run()
        except Exception as e:
            print(e)
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return

    def close(self):
        self._set_status("stopped")
        self.event.clear()

    def terminate(self):
        self.close()
        self.join()

    def wait(self):
        try:
            self.join()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self.terminate()

    @property
    def status(self):
        return self._status["status"]

    @property
    def exception(self):
        if self._parent_conn.poll():
            self._exception = self._parent_conn.recv()
        return self._exception


if __name__ == '__main__':
    """
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
    """

    server_class = "zerollama.core.framework.zero.server:Z_MethodZeroServer"
    h = ZeroServerProcess(server_class, {"do_register": False, "port": 9527})
    h.start()
    h.wait()
