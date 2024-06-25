
from gevent.threadpool import ThreadPoolExecutor
from zerollama.core.framework.zero.server import ZeroServerProcess, Z_MethodZeroServer
from zerollama.core.framework.zero_manager.protocol import ZeroServerResponseOk, ZeroServerResponseError
from zerollama.core.framework.zero_manager.protocol import StartRequest, TerminateRequest, StatusRequest


class ZeroManager(Z_MethodZeroServer):
    protocol = "manager"

    def __init__(self, name, server_class=None, **kwargs):
        super().__init__(name=name, port=None, do_register=True, **kwargs)
        self._engines = None
        self.server_class = server_class

    def init(self):
        self._engines = {}
        print(f"{self.__class__.__name__} for {self.name} running!", "port:", self.port)

    def clean_up(self):
        for k, engine in self._engines.items():
            try:
                engine.terminate()
            except Exception as e:
                pass
        super().clean_up()

    def z_start(self, req):
        kwargs = StartRequest(**req.data)

        if kwargs.name in self._engines:
            rep = ZeroServerResponseOk(msg={"already_started": True})
            self.zero_send(req, rep)
            return

        if "name" not in kwargs.engine_kwargs:
            kwargs.engine_kwargs["name"] = kwargs.name

        server_class = kwargs.engine_kwargs.pop("server_class", self.server_class)

        engine = ZeroServerProcess(server_class, kwargs.engine_kwargs)
        self._engines[kwargs.name] = engine

        with ThreadPoolExecutor(1) as executor:
            f = executor.submit(engine.start)
            f.result()
        ## engine.start() ?????

        rep = ZeroServerResponseOk(msg={"already_started": False})
        self.zero_send(req, rep)

    def z_terminate(self, req):
        kwargs = TerminateRequest(**req.data)

        if kwargs.name not in self._engines:
            rep = ZeroServerResponseOk(msg={"founded": False})
            self.zero_send(req, rep)
            return

        engine = self._engines.pop(kwargs.name)

        exception = engine.exception
        if exception is not None:
            exception = f"ZeroManager exception \n {str(exception[0])}"
        msg = {"founded": True, "last_status": engine.status, "last_exception": exception}

        engine.terminate()
        rep = ZeroServerResponseOk(msg=msg)
        self.zero_send(req, rep)

    def z_list(self, req):
        rep = ZeroServerResponseOk(msg=list(self._engines.keys()))
        self.zero_send(req, rep)

    def z_statuses(self, req):
        msg = {k: v.status for k, v in self._engines.items()}
        rep = ZeroServerResponseOk(msg=msg)
        self.zero_send(req, rep)

    def z_status(self, req):
        kwargs = StatusRequest(**req.data)
        if kwargs.name not in self._engines:
            err_msg = f"[kwargs.name] not found."
            self.handle_error(err_msg, req=req)
            return

        engine = self._engines[kwargs.name]

        exception = engine.exception

        if exception is not None:
            exception = f"ZeroManager exception \n {str(exception[0])}"

        msg = {"status": engine.status, "exception": exception}
        rep = ZeroServerResponseOk(msg=msg)
        self.zero_send(req, rep)


if __name__ == '__main__':
    name = "ZeroChatInferenceManager"
    server_class = "zerollama.tasks.chat.engine.server:ZeroChatInferenceEngine"

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    manager = ZeroServerProcess("zerollama.core.framework.zero_manager.server:ZeroManager",
                                server_kwargs={
                                    "name": name,
                                    "server_class": server_class
                                })

    nameserver.start()
    manager.start()

    manager.wait()
    nameserver.wait()