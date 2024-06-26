
from zerollama.tasks.ocr.table_structure_recognition.interface import TableStructureRecognitionModel


class DeepDocTableStructureRecognition(TableStructureRecognitionModel):
    family = "deepdoc_tsr"
    header = ["name"]
    info = [
        ["deepdoc_tsr"]
    ]
    inference_backend = "zerollama.models.ragflow.backend.table_structure_recognition:DeepDocTableStructureRecognition"


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.models.ragflow.document_layout_analysis import DeepDocDocumentLayoutAnalysis
    from zerollama.tasks.ocr.table_structure_recognition.utils import get_annotated_image

    test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/tsr"
    input_path = test_path / "PMC3863500_00003.jpg"

    image = Image.open(input_path)

    dla = DeepDocDocumentLayoutAnalysis.get_model("deepdoc_dla/layout")
    dla.load()

    layout = dla.detection(image, options={"thr": 0.3})

    table = [x for x in layout.bboxes if x.label == 'table'][0]
    table_image = image.crop(table.bbox)

    for model_name in [x[0] for x in DeepDocTableStructureRecognition.info]:
        model = DeepDocTableStructureRecognition.get_model(model_name)
        model.load()
        results = model.recognition(table_image, options={"thr": 0.2})
        annotated_image = get_annotated_image(table_image, results)
        annotated_image.save(f'result-{model_name.replace("/", "-")}.jpg')
