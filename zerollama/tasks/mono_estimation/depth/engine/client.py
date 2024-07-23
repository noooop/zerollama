
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.mono_estimation.depth.protocol import PROTOCOL
from zerollama.tasks.mono_estimation.depth.protocol import DepthEstimationRequest, DepthEstimationResponse

CLIENT_VALIDATION = True


class DepthEstimationClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def estimation(self, name, image):
        method = "inference"
        data = {"model": name,
                "image": image}
        if CLIENT_VALIDATION:
            data = DepthEstimationRequest(**data).dict()

        rep = self.query(name, method, data)
        if rep is None:
            raise RuntimeError(f"{self.__class__.__class__} [{name}] server not found.")

        if rep.state != "ok":
            raise RuntimeError(f"{self.__class__.__class__} [{name}] error, with error msg [{rep.msg}]")

        return DepthEstimationResponse(**rep.msg)


if __name__ == '__main__':
    import requests
    import numpy as np
    from PIL import Image

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    model_name = "depth-anything/Depth-Anything-V2-Small-hf"

    client = DepthEstimationClient()
    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('ZeroDepthEstimationInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    output = client.estimation(model_name, image).depth

    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    depth.save(f'result-{model_name.replace("/", "-")}.jpg')