
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.ocr.text_line_detection.protocol import PROTOCOL
from zerollama.tasks.ocr.text_line_detection.protocol import TextLineDetectionRequest, TextDetectionResult

CLIENT_VALIDATION = True


class TLDClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def detection(self, name, image, options=None):
        method = "inference"
        data = {"model": name,
                "image": image,
                "options": options or dict()}
        if CLIENT_VALIDATION:
            data = TextLineDetectionRequest(**data).dict()

        rep = self.query(name, method, data)
        if rep is None:
            raise RuntimeError(f"{self.__class__.__name__} [{name}] server not found.")

        if rep.state != "ok":
            raise RuntimeError(f"{self.__class__.__name__} [{name}] error, with error msg [{rep.msg}]")

        return TextDetectionResult(**rep.msg)


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.tasks.ocr.text_line_detection.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    image = Image.open(input_path)

    model_name = "surya_tld"

    client = TLDClient()
    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('ZeroTLDInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    results = client.detection(model_name, image)
    annotated_image = get_annotated_image(image, results)
    annotated_image.save(f'result-{model_name.replace("/", "-")}.jpg')