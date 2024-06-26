
from zerollama.tasks.ocr.reading_order_detection.interface import ReadingOrderDetectionModel


class SuryaReadingOrderDetection(ReadingOrderDetectionModel):
    family = "surya_rod"
    header = ["name"]
    info = [
        ["surya_rod"]
    ]
    inference_backend = "zerollama.models.surya.backend.reading_order_detection:SuryaReadingOrderDetection"


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.models.surya.text_line_detection import SuryaTextLineDetection
    from zerollama.models.surya.document_layout_analysis import SuryaDocumentLayoutAnalysis
    from zerollama.tasks.ocr.reading_order_detection.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    image = Image.open(input_path)

    tld = SuryaTextLineDetection.get_model("surya_tld")
    tld.load()
    lines = tld.detection(image)

    dla = SuryaDocumentLayoutAnalysis.get_model("surya_dla")
    dla.load()
    layout = dla.detection(image, lines)

    rod = SuryaReadingOrderDetection.get_model("surya_rod")
    rod.load()
    results = rod.detection(image, layout)

    annotated_image = get_annotated_image(image, results)
    annotated_image.save('result-surya_rod.jpg')



