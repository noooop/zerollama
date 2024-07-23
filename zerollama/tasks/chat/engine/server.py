
import inspect
from zerollama.tasks.chat.collection import get_model_by_name
from zerollama.tasks.chat.protocol import ChatCompletionRequest, ChatCompletionStreamResponseDone
from zerollama.tasks.chat.protocol import ZeroServerResponseOk, ZeroServerStreamResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroChatInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req, **kwargs):
        ccr = ChatCompletionRequest(**req.data)

        kwargs = ccr.model_dump()
        kwargs.pop("model", None)

        response = self.inference.chat(**kwargs)

        if not inspect.isgenerator(response):
            rep = ZeroServerResponseOk(msg=response)
            self.zero_send(req, rep)
        else:
            for rep_id, rsp in enumerate(response):
                rep = ZeroServerStreamResponseOk(msg=rsp,
                                                 snd_more=not isinstance(rsp, ChatCompletionStreamResponseDone),
                                                 rep_id=rep_id)
                self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess("zerollama.tasks.chat.engine.server:ZeroChatInferenceEngine",
                               server_kwargs={
                                   "name": "Qwen/Qwen1.5-0.5B-Chat",
                                   "engine_kwargs": {
                                       "inference_backend": "zerollama.microservices.inference.vllm_green.chat:VLLMChat"}
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()
