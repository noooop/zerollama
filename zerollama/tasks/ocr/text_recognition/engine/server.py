from zerollama.tasks.ocr.text_recognition.collection import get_model_by_name
from zerollama.tasks.ocr.text_recognition.protocol import TextRecognitionRequest
from zerollama.tasks.ocr.text_recognition.protocol import ZeroServerResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroTRInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        data = TextRecognitionRequest(**req.data)
        response = self.inference.recognition(data.image, data.lang, data.lines, data.options)
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.tasks.ocr.text_recognition.protocol import TR_ENGINE_CLASS
    from zerollama.tasks.ocr.text_line_detection.protocol import TLD_ENGINE_CLASS

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    tr_engine = ZeroServerProcess(TR_ENGINE_CLASS,
                                  server_kwargs={
                                      "name": "surya_tr",
                                      "engine_kwargs": {}
                                  })

    tld_engine = ZeroServerProcess(TLD_ENGINE_CLASS,
                                   server_kwargs={
                                       "name": "surya_tld",
                                       "engine_kwargs": {}
                                   })

    nameserver.start()
    tr_engine.start()
    tld_engine.start()

    tr_engine.wait()
    tld_engine.wait()
    nameserver.wait()
