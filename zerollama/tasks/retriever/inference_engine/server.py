
from zerollama.tasks.retriever.collection import get_model_by_name
from zerollama.tasks.retriever.protocol import RetrieverRequest
from zerollama.tasks.retriever.protocol import ZeroServerResponseOkWithPayload
from zerollama.tasks.base.inference_engine.server import ZeroInferenceEngine


class ZeroRetrieverInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        rr = RetrieverRequest(**req.data)
        response = self.inference.encode(rr.sentences, rr.options)
        rep = ZeroServerResponseOkWithPayload.load(response, tensor_field="vecs")
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



