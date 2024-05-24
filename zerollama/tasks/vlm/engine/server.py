
from zerollama.tasks.vlm.collection import get_model_by_name
from zerollama.tasks.vlm.protocol import VLMChatCompletionRequest
from zerollama.tasks.vlm.protocol import ZeroServerResponseOk, ZeroServerStreamResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroVLMInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        data = VLMChatCompletionRequest(**req.data)

        response = self.inference.chat(data.messages, data.images, data.options)
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.tasks.vlm.protocol import ENGINE_CLASS

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess(ENGINE_CLASS,
                               server_kwargs={
                                   "name": "openbmb/MiniCPM-V",
                                   "engine_kwargs": {}
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



