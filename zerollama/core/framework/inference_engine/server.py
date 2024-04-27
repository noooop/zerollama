
import json
from zerollama.core.framework.zero.server import Z_MethodZeroServer
from zerollama.core.framework.inference_engine.protocol import ChatCompletionRequest
from zerollama.core.framework.inference_engine.protocol import ZeroServerResponseOk, ZeroServerStreamResponseOk


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
    engine = ZeroServerProcess("zerollama.core.framework.inference_engine.server:ZeroInferenceEngine",
                               server_kwargs={
                                   "model_class": "zerollama.inference_backend.hf_transformers.main:HuggingFaceTransformersChat",
                                   "model_kwargs": {
                                     "model_name": "Qwen/Qwen1.5-0.5B-Chat"
                                   }
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



