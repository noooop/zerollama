
from zerollama.tasks.chat.collection import get_model_by_name
from zerollama.tasks.chat.protocol import ChatCompletionRequest, ChatCompletionStreamResponseDone
from zerollama.tasks.chat.protocol import ZeroServerResponseOk, ZeroServerStreamResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroChatInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        ccr = ChatCompletionRequest(**req.data)
        if ccr.stream:
            for rep_id, response in enumerate(self.inference.stream_chat(ccr.messages, ccr.options)):
                rep = ZeroServerStreamResponseOk(msg=response,
                                                 snd_more=not isinstance(response, ChatCompletionStreamResponseDone),
                                                 rep_id=rep_id)
                self.zero_send(req, rep)
        else:
            response = self.inference.chat(ccr.messages, ccr.options)
            rep = ZeroServerResponseOk(msg=response)
            self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess("zerollama.tasks.chat.engine.server:ZeroChatInferenceEngine",
                               server_kwargs={
                                   "model_name": "Qwen/Qwen1.5-0.5B-Chat",
                                   "model_kwargs": {}
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



