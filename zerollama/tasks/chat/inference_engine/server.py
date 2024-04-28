
from zerollama.core.framework.zero.server import Z_MethodZeroServer
from zerollama.tasks.chat.interface import ChatModel
from zerollama.tasks.chat.protocol import ChatCompletionRequest
from zerollama.tasks.chat.protocol import ZeroServerResponseOk, ZeroServerStreamResponseOk


class ZeroChatInferenceEngine(Z_MethodZeroServer):
    def __init__(self, model_name, model_kwargs, **kwargs):
        self.model_name = model_name
        self.model_class = ChatModel.inference_backend

        if isinstance(self.model_class, str):
            module_name, class_name = self.model_class.split(":")
            import importlib
            module = importlib.import_module(module_name)
            self.model_class = getattr(module, class_name)

        self.model = self.model_class(model_name=model_name, **model_kwargs)

        Z_MethodZeroServer.__init__(self, name=model_name, protocol=self.model.protocol,
                                    port=None, do_register=True, **kwargs)

    def init(self):
        self.model.load()
        print(f"{self.__class__.__name__}: ", self.name, "is running!", "port:", self.port)

    def z_inference(self, req):
        ccr = ChatCompletionRequest(**req.data)
        if ccr.stream:
            for rep_id, response in enumerate(self.model.stream_chat(ccr.messages, ccr.options)):
                rep = ZeroServerStreamResponseOk(msg=response, snd_more=not response.done, rep_id=rep_id)
                self.zero_send(req, rep)
        else:
            response = self.model.chat(ccr.messages, ccr.options)
            rep = ZeroServerResponseOk(msg=response)
            self.zero_send(req, rep)

    def z_info(self, req):
        if hasattr(self.model, "info"):
            info = self.model.info
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



