from zerollama.tasks.ocr.table_structure_recognition.collection import get_model_by_name
from zerollama.tasks.ocr.table_structure_recognition.protocol import TableStructureRecognitionRequest
from zerollama.tasks.ocr.table_structure_recognition.protocol import ZeroServerResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroTSRInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        data = TableStructureRecognitionRequest(**req.data)

        response = self.inference.recognition(data.image, data.options)
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.tasks.ocr.document_layout_analysis.protocol import DLA_ENGINE_CLASS
    from zerollama.tasks.ocr.table_structure_recognition.protocol import TSR_ENGINE_CLASS

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    dla_engine = ZeroServerProcess(DLA_ENGINE_CLASS,
                                   server_kwargs={
                                       "name": "deepdoc_dla/layout",
                                       "engine_kwargs": {}
                                   })

    tsr_engine = ZeroServerProcess(TSR_ENGINE_CLASS,
                                   server_kwargs={
                                       "name": "deepdoc_tsr",
                                       "engine_kwargs": {}
                                   })

    nameserver.start()
    dla_engine.start()
    tsr_engine.start()

    dla_engine.wait()
    tsr_engine.wait()
    nameserver.wait()
