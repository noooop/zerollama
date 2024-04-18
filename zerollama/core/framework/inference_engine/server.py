import zmq
import json
from zerollama.core.framework.zero.server import ZeroServer


class ZeroInferenceEngine(ZeroServer):
    def __init__(self, model_class, model_kwargs, event=None):
        ZeroServer.__init__(self, port=None, event=event, do_register=True)

        self.name = model_kwargs["model_name"]
        self.model_class = model_class
        self.model_kwargs = model_kwargs

        if isinstance(self.model_class, str):
            module_name, class_name = self.model_class.split(":")
            import importlib
            module = importlib.import_module(module_name)
            self.model_class = getattr(module, class_name)

        self.model = self.model_class(**self.model_kwargs)
        self.protocol = self.model_class.protocol

    def init(self):
        self.model.load()
        print("ZeroInferenceEngine: ", self.name, "is running!")

    def process(self):
        msg = self.socket.recv()
        try:
            msg = json.loads(msg)

            if "model" not in msg:
                self.handle_error(err_msg="'model' not in msg")
                return

            model = msg["model"]

            if model != self.model.model_name:
                self.handle_error(err_msg=f"model '{model}' not supported!")
                return

            messages = msg.get("messages", list())
            options = msg.get("options", dict())
            stream = msg.get("stream", True)

            if stream:
                try:
                    for content in self.model.stream_chat(messages, options):
                        response = json.dumps({
                            "model": model,
                            "content": content,
                            "done": False,
                        }).encode('utf8')
                        self.socket.send(response, zmq.SNDMORE)
                finally:
                    response = json.dumps({
                        "model": model,
                        "content": "",
                        "done": True,
                    }).encode('utf8')
                    self.socket.send(response)
            else:
                content = self.model.chat(messages, options)
                response = json.dumps({
                    "model": model,
                    "content": content,
                }).encode('utf8')

                self.socket.send(response)

        except Exception:
            self.handle_error(err_msg="ZeroInferenceEngine error")


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess, Event

    h = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess("zerollama.core.framework.inference_engine.server:ZeroInferenceEngine",
                               server_kwargs={
                                   "model_class": "zerollama.models.qwen.qwen1_5:Qwen1_5",
                                   "model_kwargs": {
                                     "model_name": "Qwen/Qwen1.5-0.5B-Chat"
                                   }
                               })

    h.start()
    engine.start()
    try:
        h.join()
        engine.join()
    except KeyboardInterrupt:
        h.terminate()
        engine.terminate()

