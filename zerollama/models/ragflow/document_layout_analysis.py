
from zerollama.tasks.ocr.document_layout_analysis.interface import DocumentLayoutAnalysisModel


class DeepDocDocumentLayoutAnalysis(DocumentLayoutAnalysisModel):
    family = "deepdoc_dla"
    header = ["name"]
    info = [
        ["deepdoc_dla/layout"],
        ["deepdoc_dla/layout.laws"],
        ["deepdoc_dla/layout.manual"],
        ["deepdoc_dla/layout.paper"],
    ]
    inference_backend = "zerollama.models.ragflow.backend.document_layout_analysis:DeepDocDocumentLayoutAnalysis"


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.tasks.ocr.document_layout_analysis.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/tsr"

    input_path = dla_test_path / "PMC3863500_00003.jpg"
    image = Image.open(input_path)

    for model_name in [x[0] for x in DeepDocDocumentLayoutAnalysis.info]:
        model = DeepDocDocumentLayoutAnalysis.get_model(model_name)
        model.load()

        results = model.detection(image, options={"thr": 0.3})
        annotated_image = get_annotated_image(image, results)
        annotated_image.save(f'result-{model_name.replace("/", "-")}.jpg')

