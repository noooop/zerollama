
import json
from zerollama.core.framework.zero.server import ZeroServerProcess, Z_MethodZeroServer
from zerollama.core.framework.zero_manager.protocol import ZeroServerResponseOk
from zerollama.core.framework.zero_manager.protocol import StartRequest, TerminateRequest


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

        engine = ZeroServerProcess(self.server_class,
                                   server_kwargs={
                                       "model_class": kwargs.model_class,
                                       "model_kwargs": kwargs.model_kwargs
                                   })
        engine.start()

        self._inference_engines[kwargs.name] = engine
        rep = ZeroServerResponseOk(msg={"already_started": False})
        self.zero_send(req, rep)

    def z_terminate(self, req):
        kwargs = TerminateRequest(**req.data)

        if kwargs.name not in self._inference_engines:
            rep = ZeroServerResponseOk(msg={"founded": False})
            self.zero_send(req, rep)
            return

        engine = self._inference_engines.pop(kwargs.name)
        engine.terminate()

        rep = ZeroServerResponseOk(msg={"founded": True})
        self.zero_send(req, rep)

    def z_list(self, req):
        rep = ZeroServerResponseOk(msg=list(self._inference_engines.keys()))
        self.zero_send(req, rep)


if __name__ == '__main__':
    name = "ZeroInferenceManager"
    server_class = "zerollama.core.framework.inference_engine.server:ZeroInferenceEngine"

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