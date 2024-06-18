
from zerollama.tasks.ocr.text_recognition.interface import TextRecognitionModel


class PaddleOCRTextRecognition(TextRecognitionModel):
    family = "PaddleOCR_tr"
    lang = 'ch'
    header = ["name", "ocr_version", "rec_algorithm"]
    info = [
        ["PaddleOCR_tr/PP-OCRv4/SVTR_LCNet", "PP-OCRv4", "SVTR_LCNet"],
        ["PaddleOCR_tr/PP-OCRv3/SVTR_LCNet", "PP-OCRv3", "SVTR_LCNet"],
        ["PaddleOCR_tr/PP-OCRv2/SVTR_LCNet", "PP-OCRv2", "SVTR_LCNet"],
        ["PaddleOCR_tr/PP-OCR/SVTR_LCNet",   "PP-OCR",   "SVTR_LCNet"],

        ["PaddleOCR_tr/PP-OCRv4/CRNN",       "PP-OCRv4", "CRNN"],
        ["PaddleOCR_tr/PP-OCRv3/CRNN",       "PP-OCRv3", "CRNN"],
        ["PaddleOCR_tr/PP-OCRv2/CRNN",       "PP-OCRv2", "CRNN"],
        ["PaddleOCR_tr/PP-OCR/CRNN",         "PP-OCR",   "CRNN"],
    ]
    inference_backend = "zerollama.models.paddleocr.backend.text_recognition:PaddleOCRTextRecognition"


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.models.paddleocr.text_line_detection import PaddleOCRTextLineDetection
    from zerollama.tasks.ocr.text_recognition.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    image = Image.open(input_path)

    tld = PaddleOCRTextLineDetection.get_model("PaddleOCR_tld/PP-OCRv4")
    tld.load()
    lines = tld.detection(image)

    langs = ["zh", "en"]

    for model_name in [x[0] for x in PaddleOCRTextRecognition.info]:
        model = PaddleOCRTextRecognition.get_model(model_name)
        model.load()
        results = model.recognition(image, langs, lines)

        annotated_image = get_annotated_image(image, results, langs)
        annotated_image.save(f'result-{model_name.replace("/", "-")}.jpg')

