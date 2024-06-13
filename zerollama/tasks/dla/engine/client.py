
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.dla.protocol import PROTOCOL
from zerollama.tasks.dla.protocol import DLARequest

CLIENT_VALIDATION = True


class DLAClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def detection(self, name, image, options=None):
        method = "inference"
        data = {"model": name,
                "image": image,
                "options": options or dict(),
                "stream": False}
        if CLIENT_VALIDATION:
            data = DLARequest(**data).dict()

        rep = self.query(name, method, data)
        if rep is None:
            raise RuntimeError(f"DLA [{name}] server not found.")

        if rep.state != "ok":
            raise RuntimeError(f"DLA [{name}] error, with error msg [{rep.msg}]")

        return rep.msg


if __name__ == '__main__':
    import os
    import cv2
    from pathlib import Path
    from zerollama.tasks.dla.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    img_BGR = cv2.imread(str(input_path))
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    model_name = "YOLOv10-Document-Layout-Analysis/yolov10x_best.pt"

    client = DLAClient()
    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('ZeroDLAInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    results = client.detection(model_name, img_RGB, options={"conf": 0.2, "iou": 0.8})
    annotated_image = get_annotated_image(img_RGB, results)
    cv2.imwrite(f'result-{model_name.replace("/", "-")}.jpg', annotated_image)