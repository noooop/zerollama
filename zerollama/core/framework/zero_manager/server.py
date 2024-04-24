
import json
from zerollama.core.framework.zero.server import ZeroServerProcess, Z_MethodZeroServer


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

    def z_start(self, uuid, msg):
        if "model_class" not in msg:
            err_msg = "'model_class' not in msg"
            self.handle_error(uuid, err_msg)
            return
        model_class = msg['model_class']

        if "model_kwargs" not in msg:
            err_msg = "'model_kwargs' not in msg"
            self.handle_error(uuid, err_msg)
            return
        model_kwargs = msg['model_kwargs']

        if "name" not in msg:
            err_msg = "'name' not in msg"
            self.handle_error(uuid, err_msg)
            return

        name = msg["name"]

        if name in self._inference_engines:
            response = json.dumps({
                "state": "ok",
                "msg": {"already_started": True}
            }).encode('utf8')
            self.socket.send_multipart(uuid+[response])
            return

        try:
            engine = ZeroServerProcess(self.server_class,
                                       server_kwargs={
                                           "model_class": model_class,
                                           "model_kwargs": model_kwargs
                                       })
            engine.start()
        except Exception as e:
            self.handle_error(uuid, err_msg=str(e))
            return

        self._inference_engines[name] = engine
        response = json.dumps({
            "state": "ok",
            "msg": {"already_started": False}
        }).encode('utf8')
        self.socket.send_multipart(uuid+[response])

    def z_terminate(self, uuid, msg):
        if "name" not in msg:
            err_msg = "'name' not in msg"
            self.handle_error(uuid, err_msg)
            return
        name = msg["name"]

        if name not in self._inference_engines:
            response = json.dumps({
                "state": "ok",
                "msg": {"founded": False}
            }).encode('utf8')
            self.socket.send_multipart(uuid+[response])
            return

        try:
            engine = self._inference_engines.pop(name)
            engine.terminate()
        except Exception as e:
            self.handle_error(uuid, err_msg=str(e))
            return

        response = json.dumps({
            "state": "ok",
            "msg": {"founded": True}
        }).encode('utf8')
        self.socket.send_multipart(uuid+[response])

    def z_list(self, uuid, msg):
        response = json.dumps({
            "state": "ok",
            "msg": list(self._inference_engines.keys())
        }).encode('utf8')
        self.socket.send_multipart(uuid+[response])


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