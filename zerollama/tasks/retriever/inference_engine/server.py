
from zerollama.core.framework.zero.server import Z_MethodZeroServer
from zerollama.tasks.retriever.collection import get_model_by_name
from zerollama.tasks.retriever.protocol import RetrieverRequest
from zerollama.tasks.retriever.protocol import ZeroServerResponseOk, ZeroServerResponseOkWithPayload


class ZeroRetrieverInferenceEngine(Z_MethodZeroServer):
    def __init__(self, model_name, model_kwargs, **kwargs):
        self.model_name = model_name
        self.model_class = get_model_by_name(model_name)
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
        rr = RetrieverRequest(**req.data)
        response = self.inference.encode(rr.sentences, rr.options)

        rep = ZeroServerResponseOkWithPayload.combine(response, tensor_field="vecs")
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
    engine = ZeroServerProcess("zerollama.tasks.retriever.inference_engine.server:ZeroRetrieverInferenceEngine",
                               server_kwargs={
                                   "model_name": "BAAI/bge-m3",
                                   "model_kwargs": {}
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



