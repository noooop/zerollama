
from zerollama.tasks.retriever.collection import get_model_by_name
from zerollama.tasks.retriever.protocol import RetrieverRequest
from zerollama.tasks.retriever.protocol import ZeroServerResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroRetrieverInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        rr = RetrieverRequest(**req.data)
        response = self.inference.encode(rr.sentences, rr.options)
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess("zerollama.tasks.retriever.engine.server:ZeroRetrieverInferenceEngine",
                               server_kwargs={
                                   "name": "BAAI/bge-m3",
                                   "engine_kwargs": {}
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



