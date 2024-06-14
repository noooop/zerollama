
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.ocr.text_recognition.protocol import PROTOCOL
from zerollama.tasks.ocr.text_recognition.protocol import TextRecognitionRequest, TextRecognitionResult

CLIENT_VALIDATION = True


class TRClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def recognition(self, name, image, lang, bboxes, options=None):
        method = "inference"
        data = {"model": name,
                "image": image,
                "lang": lang,
                "bboxes": bboxes,
                "options": options or dict()}
        if CLIENT_VALIDATION:
            data = TextRecognitionRequest(**data).dict()

        rep = self.query(name, method, data)
        if rep is None:
            raise RuntimeError(f"{self.__class__.__name__} [{name}] server not found.")

        if rep.state != "ok":
            raise RuntimeError(f"{self.__class__.__name__} [{name}] error, with error msg [{rep.msg}]")

        return TextRecognitionResult(**rep.msg)


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.tasks.ocr.text_line_detection.engine.client import TLDClient

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    image = Image.open(input_path)

    tld_model_name = "surya_tld"

    tld_client = TLDClient()
    tld_client.wait_service_available(tld_model_name)
    results = tld_client.detection(tld_model_name, image)

    model_name = "surya_tr"
    client = TRClient()
    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('ZeroTRInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    results = client.recognition(model_name, image, ["zh", "en"], [{"bbox": x.bbox} for x in results.bboxes])

    for line in results.text_lines:
        print(line)