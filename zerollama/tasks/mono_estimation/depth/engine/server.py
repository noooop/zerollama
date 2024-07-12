
from zerollama.tasks.mono_estimation.depth.collection import get_model_by_name
from zerollama.tasks.mono_estimation.depth.protocol import DepthEstimationRequest
from zerollama.tasks.mono_estimation.depth.protocol import ZeroServerResponseOk
from zerollama.tasks.base.engine.server import ZeroInferenceEngine


class ZeroDepthEstimationInferenceEngine(ZeroInferenceEngine):
    get_model_by_name = staticmethod(get_model_by_name)

    def inference_worker(self, req):
        data = DepthEstimationRequest(**req.data)

        response = self.inference.estimation(data.image)
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.tasks.mono_estimation.depth.protocol import ENGINE_CLASS

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess(ENGINE_CLASS,
                               server_kwargs={
                                   "name": "depth-anything/Depth-Anything-V2-Small-hf",
                                   "engine_kwargs": {}
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



