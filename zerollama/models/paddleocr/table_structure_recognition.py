
from zerollama.tasks.ocr.table_structure_recognition.interface import TableStructureRecognitionModel


class PaddleOCRTableStructureRecognition(TableStructureRecognitionModel):
    family = "PaddleOCR_tsr"
    header = ["name", "structure_version", "lang"]
    info = [
        ["PaddleOCR_tsr/PP-Structure",   "PP-Structure",   "en"],
        ["PaddleOCR_tsr/PP-StructureV2", "PP-StructureV2", "ch"],
    ]
    inference_backend = "zerollama.models.paddleocr.backend.table_structure_recognition:PaddleOCRTableStructureRecognition"


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.models.paddleocr.document_layout_analysis import PaddleOCRDocumentLayoutAnalysis
    from zerollama.tasks.ocr.table_structure_recognition.utils import get_annotated_image

    test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/tsr"
    input_path = test_path / "PMC3863500_00003.jpg"

    image = Image.open(input_path)

    dla = PaddleOCRDocumentLayoutAnalysis.get_model("PaddleOCR_dla/PP-StructureV2")
    dla.load()

    layout = dla.detection(image)

    table = [x for x in layout.bboxes if x.label == 'table'][0]
    table_image = image.crop(table.bbox)

    for model_name in [x[0] for x in PaddleOCRTableStructureRecognition.info]:
        model = PaddleOCRTableStructureRecognition.get_model(model_name)
        model.load()
        results = model.recognition(table_image, options={"thr": 0.2})
        annotated_image = get_annotated_image(table_image, results)
        annotated_image.save(f'result-{model_name.replace("/", "-")}.jpg')
