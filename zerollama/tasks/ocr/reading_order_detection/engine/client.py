
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.ocr.reading_order_detection.protocol import PROTOCOL
from zerollama.tasks.ocr.reading_order_detection.protocol import ReadingOrderDetectionRequest, ReadingOrderDetectionResult

CLIENT_VALIDATION = True


class RODClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def detection(self, name, image, layout, options=None):
        method = "inference"
        data = {"model": name,
                "image": image,
                "layout": layout,
                "options": options}
        if CLIENT_VALIDATION:
            data = ReadingOrderDetectionRequest(**data).dict()

        rep = self.query(name, method, data)
        if rep is None:
            raise RuntimeError(f"{self.__class__.__name__} [{name}] server not found.")

        if rep.state != "ok":
            raise RuntimeError(f"{self.__class__.__name__} [{name}] error, with error msg [{rep.msg}]")

        return ReadingOrderDetectionResult(**rep.msg)


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.tasks.ocr.text_line_detection.engine.client import TLDClient
    from zerollama.tasks.ocr.document_layout_analysis.engine.client import DLAClient
    from zerollama.tasks.ocr.reading_order_detection.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    image = Image.open(input_path)

    tld_model_name = "surya_tld"

    tld_client = TLDClient()
    tld_client.wait_service_available(tld_model_name)
    lines = tld_client.detection(tld_model_name, image)

    dla_model_name = "surya_dla"
    dla_client = DLAClient()
    layout = dla_client.detection(dla_model_name, image, lines)

    model_name = "surya_rod"

    client = RODClient()

    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('ZeroRODInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    results = client.detection(model_name, image, layout)
    annotated_image = get_annotated_image(image, results)
    annotated_image.save(f'result-{model_name.replace("/", "-")}.jpg')