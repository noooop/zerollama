
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.super_resolution.protocol import PROTOCOL
from zerollama.tasks.super_resolution.protocol import SRRequest, SRResponse

CLIENT_VALIDATION = True


class SRClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def sr(self, name, image, options=None):
        method = "inference"
        data = {"model": name,
                "image": image,
                "options": options or dict()}

        if CLIENT_VALIDATION:
            data = SRRequest(**data).dict()

        rep = self.query(name, method, data)
        if rep is None:
            raise RuntimeError(f"super resolution [{name}] server not found.")

        if rep.state == "ok":
            rep = SRResponse(**rep.msg)
        else:
            raise RuntimeError(f"super resolution [{name}] error, with error msg [{rep.msg}]")
        return rep


if __name__ == '__main__':
    import os
    import cv2
    from PIL import Image
    from pathlib import Path

    CLIENT_VALIDATION = False

    model_name = "4xGRL"

    client = SRClient()
    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('ZeroSRInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    vlm_test_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent / "static/vlm_test"
    input_path = vlm_test_path / "monday.jpg"
    img_lr = cv2.imread(str(input_path))
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)

    response = client.sr(model_name, img_lr)

    print(f"input shape {img_lr.shape}, output shape {response.image.shape}")
