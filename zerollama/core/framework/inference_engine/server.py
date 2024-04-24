import zmq
import json
from zerollama.core.framework.zero.server import Z_MethodZeroServer


class ZeroInferenceEngine(Z_MethodZeroServer):
    def __init__(self, model_class, model_kwargs, **kwargs):
        self.model_class = model_class
        self.model_kwargs = model_kwargs

        if isinstance(self.model_class, str):
            module_name, class_name = self.model_class.split(":")
            import importlib
            module = importlib.import_module(module_name)
            self.model_class = getattr(module, class_name)

        self.model = self.model_class(**self.model_kwargs)

        Z_MethodZeroServer.__init__(self,
                                    name=model_kwargs["model_name"],
                                    protocol=self.model_class.protocol,
                                    port=None, do_register=True, **kwargs)

    def init(self):
        self.model.load()
        print("ZeroInferenceEngine: ", self.name, "is running!", "port:", self.port)

    def z_inference(self, uuid, msg):
        if "model" not in msg:
            self.handle_error(uuid, err_msg="'model' not in msg")
            return

        model = msg["model"]

        if model != self.model.model_name:
            self.handle_error(uuid, err_msg=f"model '{model}' not supported!")
            return

        messages = msg.get("messages", list())
        options = msg.get("options", dict())
        stream = msg.get("stream", True)

        try:
            if stream:
                for rep in self.model.stream_chat(messages, options):
                    response = json.dumps(rep).encode('utf8')
                    self.socket.send_multipart(uuid + [response])
            else:
                rep = self.model.chat(messages, options)
                response = json.dumps(rep).encode('utf8')
                self.socket.send_multipart(uuid + [response])
        except Exception:
            self.handle_error(uuid, err_msg="ZeroInferenceEngine error")


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess("zerollama.core.framework.inference_engine.server:ZeroInferenceEngine",
                               server_kwargs={
                                   "model_class": "zerollama.models.qwen.qwen1_5:Qwen1_5",
                                   "model_kwargs": {
                                     "model_name": "Qwen/Qwen1.5-0.5B-Chat"
                                   }
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



