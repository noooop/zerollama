
from gevent.threadpool import ThreadPoolExecutor
from zerollama.core.framework.zero.server import ZeroServerProcess, Z_MethodZeroServer
from zerollama.core.framework.zero_manager.protocol import ZeroServerResponseOk, ZeroServerResponseError
from zerollama.core.framework.zero_manager.protocol import StartRequest, TerminateRequest, StatusRequest


class ZeroManager(Z_MethodZeroServer):
    protocol = "manager"

    def __init__(self, name, server_class, **kwargs):
        super().__init__(name=name, port=None, do_register=True, **kwargs)
        self._inference_engines = None
        self.server_class = server_class
        if isinstance(server_class, str):
            self.module_name, self.class_name = server_class.split(":")

    def init(self):
        self._inference_engines = {}
        print(f"{self.__class__.__name__} for {self.class_name} running!", "port:", self.port)

    def clean_up(self):
        for k, engine in self._inference_engines.items():
            try:
                engine.terminate()
            except Exception as e:
                pass
        super().clean_up()

    def z_start(self, req):
        kwargs = StartRequest(**req.data)

        if kwargs.name in self._inference_engines:
            rep = ZeroServerResponseOk(msg={"already_started": True})
            self.zero_send(req, rep)
            return

        server_kwargs = {"model_name": kwargs.name,
                         "model_kwargs": kwargs.model_kwargs}

        engine = ZeroServerProcess(self.server_class, server_kwargs)
        self._inference_engines[kwargs.name] = engine

        with ThreadPoolExecutor(1) as executor:
            f = executor.submit(engine.start)
            f.result()
        ## engine.start() ?????

        rep = ZeroServerResponseOk(msg={"already_started": False})
        self.zero_send(req, rep)

    def z_terminate(self, req):
        kwargs = TerminateRequest(**req.data)

        if kwargs.name not in self._inference_engines:
            rep = ZeroServerResponseOk(msg={"founded": False})
            self.zero_send(req, rep)
            return

        engine = self._inference_engines.pop(kwargs.name)

        exception = engine.exception
        if exception is not None:
            exception = f"{self.class_name}: {str(exception[0])}"
        msg = {"founded": True, "last_status": engine.status, "last_exception": exception}

        engine.terminate()
        rep = ZeroServerResponseOk(msg=msg)
        self.zero_send(req, rep)

    def z_list(self, req):
        rep = ZeroServerResponseOk(msg=list(self._inference_engines.keys()))
        self.zero_send(req, rep)

    def z_statuses(self, req):
        msg = {k: v.status for k, v in self._inference_engines.items()}
        rep = ZeroServerResponseOk(msg=msg)
        self.zero_send(req, rep)

    def z_status(self, req):
        kwargs = StatusRequest(**req.data)
        if kwargs.name not in self._inference_engines:
            err_msg = f"[kwargs.name] not found."
            self.handle_error(err_msg, req=req)
            return

        engine = self._inference_engines[kwargs.name]

        exception = engine.exception

        if exception is not None:
            exception = f"{self.class_name}: {str(exception[0])}"

        msg = {"status": engine.status, "exception": exception}
        rep = ZeroServerResponseOk(msg=msg)
        self.zero_send(req, rep)


if __name__ == '__main__':
    name = "ZeroChatInferenceManager"
    server_class = "zerollama.tasks.chat.inference_engine.server:ZeroChatInferenceEngine"

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