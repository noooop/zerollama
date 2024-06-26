from zerollama.tasks.ocr.reading_order_detection.collection import get_model_by_name
from zerollama.tasks.ocr.reading_order_detection.protocol import ReadingOrderDetectionRequest
from zerollama.tasks.ocr.reading_order_detection.protocol import ZeroServerResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroRODInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        data = ReadingOrderDetectionRequest(**req.data)
        response = self.inference.detection(data.image, data.layout, data.options)
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.tasks.ocr.document_layout_analysis.protocol import DLA_ENGINE_CLASS
    from zerollama.tasks.ocr.text_line_detection.protocol import TLD_ENGINE_CLASS
    from zerollama.tasks.ocr.reading_order_detection.protocol import ROD_ENGINE_CLASS

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")

    tld_engine = ZeroServerProcess(TLD_ENGINE_CLASS,
                                   server_kwargs={
                                       "name": "surya_tld",
                                       "engine_kwargs": {}
                                   })

    dla_engine = ZeroServerProcess(DLA_ENGINE_CLASS,
                                   server_kwargs={
                                       "name": "surya_dla",
                                       "engine_kwargs": {}
                                   })

    rod_engine = ZeroServerProcess(ROD_ENGINE_CLASS,
                                  server_kwargs={
                                      "name": "surya_rod",
                                      "engine_kwargs": {}
                                  })

    nameserver.start()
    tld_engine.start()
    dla_engine.start()
    rod_engine.start()

    tld_engine.wait()
    dla_engine.wait()
    rod_engine.wait()
    nameserver.wait()
