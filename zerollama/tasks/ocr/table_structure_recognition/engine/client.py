
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.ocr.table_structure_recognition.protocol import PROTOCOL
from zerollama.tasks.ocr.table_structure_recognition.protocol import TableStructureRecognitionResult, TableStructureRecognitionRequest

CLIENT_VALIDATION = True


class TSRClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def recognition(self, name, image, options=None):
        method = "inference"
        data = {"model": name,
                "image": image,
                "options": options or dict()}
        if CLIENT_VALIDATION:
            data = TableStructureRecognitionRequest(**data).dict()

        rep = self.query(name, method, data)
        if rep is None:
            raise RuntimeError(f"{self.__class__.__name__} [{name}] server not found.")

        if rep.state != "ok":
            raise RuntimeError(f"{self.__class__.__name__} [{name}] error, with error msg [{rep.msg}]")

        return TableStructureRecognitionResult(**rep.msg)


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.tasks.ocr.document_layout_analysis.engine.client import DLAClient
    from zerollama.tasks.ocr.table_structure_recognition.utils import get_annotated_image

    test_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent / "static/test_sample/tsr"
    input_path = test_path / "PMC3863500_00003.jpg"

    image = Image.open(input_path)

    dla_model_name = "deepdoc_dla/layout"

    dla_client = DLAClient()
    dla_client.wait_service_available(dla_model_name)
    layout = dla_client.detection(dla_model_name, image)

    table = [x for x in layout.bboxes if x.label == 'table'][0]
    table_image = image.crop(table.bbox)

    model_name = "deepdoc_tsr"

    client = TSRClient()
    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('ZeroTSRInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    results = client.recognition(model_name, table_image)
    annotated_image = get_annotated_image(table_image, results)
    annotated_image.save(f'result-{model_name.replace("/", "-")}.jpg')
