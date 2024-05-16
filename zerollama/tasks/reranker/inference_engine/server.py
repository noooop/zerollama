
from zerollama.tasks.reranker.collection import get_model_by_name
from zerollama.tasks.reranker.protocol import RerankerRequest
from zerollama.tasks.reranker.protocol import ZeroServerResponseOkWithPayload
from zerollama.tasks.base.inference_engine.server import ZeroInferenceEngine


class ZeroRetrieverInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        data = RerankerRequest(**req.data)
        response = self.inference.compute_score(data.sentence_pairs, data.options)
        rep = ZeroServerResponseOkWithPayload.load(response, tensor_field="vecs")
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess("zerollama.tasks.reranker.inference_engine.server:ZeroRetrieverInferenceEngine",
                               server_kwargs={
                                   "model_name": "BAAI/bge-reranker-v2-m3",
                                   "model_kwargs": {}
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



