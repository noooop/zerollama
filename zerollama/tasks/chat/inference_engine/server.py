
from zerollama.core.framework.zero.server import Z_MethodZeroServer
from zerollama.tasks.chat.interface import ChatModel
from zerollama.tasks.chat.protocol import ChatCompletionRequest
from zerollama.tasks.chat.protocol import ZeroServerResponseOk, ZeroServerStreamResponseOk


class ZeroChatInferenceEngine(Z_MethodZeroServer):
    def __init__(self, model_name, model_kwargs, **kwargs):
        self.model_name = model_name
        self.model_class = ChatModel
        self.inference_backend = self.model_class.inference_backend

        if isinstance(self.inference_backend, str):
            module_name, class_name = self.inference_backend.split(":")
            import importlib
            module = importlib.import_module(module_name)
            self.inference_backend = getattr(module, class_name)

        self.inference = self.inference_backend(model_name=model_name, **model_kwargs)

        Z_MethodZeroServer.__init__(self, name=model_name, protocol=self.inference.protocol,
                                    port=None, do_register=True, **kwargs)

    def init(self):
        self.inference.load()
        print(f"{self.__class__.__name__}: ", self.name, "is running!", "port:", self.port)

    def z_inference(self, req):
        ccr = ChatCompletionRequest(**req.data)
        if ccr.stream:
            for rep_id, response in enumerate(self.inference.stream_chat(ccr.messages, ccr.options)):
                rep = ZeroServerStreamResponseOk(msg=response, snd_more=not response.done, rep_id=rep_id)
                self.zero_send(req, rep)
        else:
            response = self.inference.chat(ccr.messages, ccr.options)
            rep = ZeroServerResponseOk(msg=response)
            self.zero_send(req, rep)

    def z_info(self, req):
        if hasattr(self.inference, "info"):
            info = self.inference.info
        else:
            info = {}

        rep = ZeroServerResponseOk(msg=info)
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess("zerollama.tasks.chat.inference_engine.server:ZeroChatInferenceEngine",
                               server_kwargs={
                                   "model_name": "Qwen/Qwen1.5-0.5B-Chat",
                                   "model_kwargs": {}
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



