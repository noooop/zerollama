from zerollama.tasks.ocr.document_layout_analysis.collection import get_model_by_name
from zerollama.tasks.ocr.document_layout_analysis.protocol import DocumentLayoutAnalysisRequest
from zerollama.tasks.ocr.document_layout_analysis.protocol import ZeroServerResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroDLAInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        data = DocumentLayoutAnalysisRequest(**req.data)

        response = self.inference.detection(data.image, data.lines, data.options)
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.tasks.ocr.document_layout_analysis.protocol import DLA_ENGINE_CLASS
    from zerollama.tasks.ocr.text_line_detection.protocol import TLD_ENGINE_CLASS

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    dla_engine = ZeroServerProcess(DLA_ENGINE_CLASS,
                                   server_kwargs={
                                       "name": "surya_dla",
                                       "engine_kwargs": {}
                                   })

    tld_engine = ZeroServerProcess(TLD_ENGINE_CLASS,
                                   server_kwargs={
                                       "name": "surya_tld",
                                       "engine_kwargs": {}
                                   })

    nameserver.start()
    dla_engine.start()
    tld_engine.start()

    dla_engine.wait()
    tld_engine.wait()
    nameserver.wait()
