
from zerollama.tasks.ocr.document_layout_analysis.collection import get_model_by_name
from zerollama.tasks.ocr.document_layout_analysis.protocol import DocumentLayoutAnalysisRequest
from zerollama.tasks.ocr.document_layout_analysis.protocol import ZeroServerResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroDLAInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        data = DocumentLayoutAnalysisRequest(**req.data)

        response = self.inference.detection(data.image, data.options)
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.tasks.ocr.document_layout_analysis.protocol import ENGINE_CLASS

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess(ENGINE_CLASS,
                               server_kwargs={
                                   "name": "surya_dla",
                                   "engine_kwargs": {}
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



