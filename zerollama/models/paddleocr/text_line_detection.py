from easydict import EasyDict as edict
from zerollama.tasks.ocr.text_line_detection.interface import TextLineDetectionModel


class PaddleOCRTextLineDetection(TextLineDetectionModel):
    family = "PaddleOCR_tld"

    DB_parmas = edict({
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.6,
        "det_db_unclip_ratio": 1.5,
        "max_batch_size": 10,
        "use_dilation": False,
        "det_db_score_mode": "fast",
    })
    lang = 'ch'
    header = ["name", "ocr_version", "det_algorithm"]
    info = [
        ["PaddleOCR_tld/PP-OCRv4", "PP-OCRv4", "DB"],
        ["PaddleOCR_tld/PP-OCRv3", "PP-OCRv3", "DB"],
        ["PaddleOCR_tld/PP-OCRv2", "PP-OCRv2", "DB"],
        ["PaddleOCR_tld/PP-OCR",   "PP-OCR",   "DB"],
    ]
    inference_backend = "zerollama.models.paddleocr.backend.text_line_detection:PaddleOCRTextLineDetection"


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.tasks.ocr.text_line_detection.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    image = Image.open(input_path)

    for model_name in [x[0] for x in PaddleOCRTextLineDetection.info]:
        model = PaddleOCRTextLineDetection.get_model(model_name)
        model.load()
        results = model.detection(image)

        annotated_image = get_annotated_image(image, results)
        annotated_image.save(f'result-{model_name.replace("/", "-")}.jpg')
