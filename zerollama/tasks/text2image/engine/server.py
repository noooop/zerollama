
from zerollama.tasks.text2image.collection import get_model_by_name
from zerollama.tasks.text2image.protocol import Text2ImageRequest
from zerollama.tasks.text2image.protocol import ZeroServerResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroText2ImageInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        data = Text2ImageRequest(**req.data)

        response = self.inference.generate(data.prompt, data.negative_prompt, data.options)
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.tasks.text2image.protocol import ENGINE_CLASS

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess(ENGINE_CLASS,
                               server_kwargs={
                                   "name": "Tencent-Hunyuan/HunyuanDiT-Diffusers",
                                   "engine_kwargs": {}
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



