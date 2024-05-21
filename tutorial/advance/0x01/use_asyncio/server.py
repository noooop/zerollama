
import signal
import json
import time
import traceback
import asyncio
import zmq
import platform
from zmq.asyncio import Context
from multiprocessing import Event, Process, Value, Pipe, Manager
from zerollama.core.framework.zero.protocol import (
    Timeout,
    convert_errors,
    ValidationError,
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseOkWithPayload,
    ZeroServerResponseError
)

plat = platform.system().lower()
if plat == 'windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class ZeroServer(object):
    POLL_INTERVAL = 1000
    POOL_SIZE = 64

    def __init__(self, name=None, protocol=None, port=None, event=None, do_register=True, share_port=None,
                 nameserver_port=None):
        context = Context.instance()
        socket = context.socket(zmq.ROUTER)
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

    async def process(self, msg):
        try:
            uuid, req_id, msg, *payload = msg
        except Exception:
            traceback.print_exc()
            return

        await self.socket.send_multipart([uuid, req_id, b"ok"])

    def start(self):
        self.init()
        self.run()

    def run(self):
        if self.address_in_use:
            print(f"Address in use (addr='tcp://*:{self.port}')")
            return

        if self.do_register:
            self._register()

        class GracefulExit(SystemExit):
            code = 1

        def raise_graceful_exit(*args):
            loop = asyncio.get_event_loop()
            tasks = asyncio.all_tasks(loop=loop)
            for t in tasks:
                t.cancel()

            loop.stop()
            raise GracefulExit()

        signal.signal(signal.SIGINT, raise_graceful_exit)
        signal.signal(signal.SIGTERM, raise_graceful_exit)
        # https://stackoverflow.com/questions/45987985/asyncio-loops-add-signal-handler-in-windows
        # Windows does not have signals.

        async def _run():
            while True:
                try:
                    msg = await self.socket.recv_multipart()
                except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
                    break
                asyncio.create_task(self.process(msg))

        loop = asyncio.get_event_loop()
        task = loop.create_task(_run())
        try:
            loop.run_until_complete(task)
        except GracefulExit:
            pass
        if 1:
            tasks = asyncio.all_tasks(loop=loop)
            for t in tasks:
                t.cancel()
            group = asyncio.gather(*tasks, return_exceptions=True)
            loop.run_until_complete(group)
            loop.close()

        if self.do_register:
            try:
                self._deregister()
            except Timeout:
                pass

        self.clean_up()

    async def handle_error(self, err_msg, uuid=None, req_id=None, req=None):
        if req is not None:
            uuid, req_id = req.uuid, req.req_id

        if isinstance(err_msg, ValidationError):
            err_msg = convert_errors(err_msg)

        response = ZeroServerResponseError(msg=err_msg)
        await self.socket.send_multipart([uuid, req_id, response.b])

    async def zero_send(self, req: ZeroServerRequest, rep: ZeroServerResponse):
        if isinstance(rep, ZeroServerStreamResponseOk):
            rep_id = req.req_id + (b"M" if rep.snd_more else b"N") + str(rep.rep_id).encode("utf-8")
        else:
            rep_id = req.req_id

        await self.socket.send_multipart([req.uuid, rep_id] + rep.b)


class Z_MethodZeroServer(ZeroServer):
    async def process(self, msg):
        try:
            uuid, req_id, msg, *payload = msg
        except Exception:
            traceback.print_exc()
            return

        try:
            msg = json.loads(msg)
        except json.JSONDecodeError as e:
            await self.handle_error(str(e), uuid, req_id)
            return

        try:
            req = ZeroServerRequest(**msg)
        except ValidationError as e:
            await self.handle_error(e, uuid, req_id)
            return

        req.uuid = uuid
        req.req_id = req_id
        req.payload = payload

        method_name = msg["method"]
        method = getattr(self, "z_" + method_name, self.default)

        try:
            await method(req)
        except ValidationError as e:
            await self.handle_error(e, req=req)
        except Exception as e:
            traceback.print_exc()
            await self.handle_error(str(e), req=req)

    async def default(self, req: ZeroServerRequest):
        method_name = req.method
        err_msg = f"method [{method_name}] not supported."
        await self.handle_error(err_msg, req=req)

    async def z_support_methods(self, req: ZeroServerRequest):
        support_methods = [m[2:] for m in dir(self) if m.startswith('z_')]
        rep = ZeroServerResponseOk(msg={"support_methods": support_methods})
        await self.zero_send(req, rep)

    async def z_info(self, req: ZeroServerRequest):
        rep = ZeroServerResponseOk(msg={})
        await self.zero_send(req, rep)


class ZeroServerProcess(Process):
    """
    Class which returns child Exceptions to Parent.
    https://stackoverflow.com/a/33599967/4992248

    status:
    prepare -> started -> initial -> running -> error or stopped
    """

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

        self._status = Manager().dict()
        self._parent_conn, self._child_conn = Pipe()
        self._exception = None

        self._set_status("prepare")

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
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return
        except Exception as e:
            traceback.print_exc()
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return

        self._set_status("initial")

        try:
            self.server.init()
        except (FileNotFoundError, EnvironmentError) as e:
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return
        except Exception as e:
            traceback.print_exc()
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return

        self._set_status("running")

        try:
            self.server.run()
        except Exception as e:
            self._set_status("error")
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            return

    def terminate(self):
        self._set_status("stopped")
        super().terminate()
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