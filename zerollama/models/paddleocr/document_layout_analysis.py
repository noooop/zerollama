
from zerollama.tasks.ocr.document_layout_analysis.interface import DocumentLayoutAnalysisModel


class PaddleOCRDocumentLayoutAnalysis(DocumentLayoutAnalysisModel):
    family = "PaddleOCR_dla"
    lang = 'ch'
    header = ["name", "structure_version"]
    info = [
        ["PaddleOCR_dla/PP-StructureV2", "PP-StructureV2"],
    ]
    inference_backend = "zerollama.models.paddleocr.backend.document_layout_analysis:PaddleOCRDocumentLayoutAnalysis"


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.tasks.ocr.document_layout_analysis.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/tsr"

    input_path = dla_test_path / "PMC3863500_00003.jpg"
    image = Image.open(input_path)

    for model_name in [x[0] for x in PaddleOCRDocumentLayoutAnalysis.info]:
        model = PaddleOCRDocumentLayoutAnalysis.get_model(model_name)
        model.load()

        results = model.detection(image)
        annotated_image = get_annotated_image(image, results)
        annotated_image.save(f'result-{model_name.replace("/", "-")}.jpg')

