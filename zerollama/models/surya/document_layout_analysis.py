
from zerollama.tasks.ocr.document_layout_analysis.interface import DocumentLayoutAnalysisModel


class SuryaDocumentLayoutAnalysis(DocumentLayoutAnalysisModel):
    family = "surya_dla"
    header = ["name"]
    info = [
        ["surya_dla"]
    ]
    inference_backend = "zerollama.models.surya.backend.document_layout_analysis:SuryaDocumentLayoutAnalysis"


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.tasks.ocr.document_layout_analysis.utils import get_annotated_image
    from zerollama.models.surya.text_line_detection import SuryaTextLineDetection

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    image = Image.open(input_path)

    tld = SuryaTextLineDetection.get_model("surya_tld")
    tld.load()
    lines = tld.detection(image)

    for model_name in [x[0] for x in SuryaDocumentLayoutAnalysis.info]:
        model = SuryaDocumentLayoutAnalysis.get_model(model_name)
        model.load()

        results = model.detection(image, lines=None)
        annotated_image = get_annotated_image(image, results)
        annotated_image.save(f'result-1-{model_name.replace("/", "-")}.jpg')

        results = model.detection(image, lines=lines)
        annotated_image = get_annotated_image(image, results)
        annotated_image.save(f'result-2-{model_name.replace("/", "-")}.jpg')

