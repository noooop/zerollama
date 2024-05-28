
from zerollama.tasks.super_resolution.collection import get_model_by_name
from zerollama.tasks.super_resolution.protocol import SRRequest
from zerollama.tasks.super_resolution.protocol import ZeroServerResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroSRInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        data = SRRequest(**req.data)
        response = self.inference.sr(data.image, **data.options)
        rep = ZeroServerResponseOk(msg=response.dict())
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.tasks.super_resolution.protocol import SR_ENGINE_CLASS

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess(SR_ENGINE_CLASS, server_kwargs={"name": "4xGRL"})

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



