from PIL import Image
from zerollama.tasks.ocr.text_recognition.interface import TextRecognitionInterface
from zerollama.tasks.ocr.text_recognition.protocol import TextLine, TextRecognitionResult
from zerollama.tasks.ocr.text_line_detection.protocol import TextDetectionResult
from zerollama.tasks.ocr.text_recognition.collection import get_model_by_name


class SuryaTextRecognition(TextRecognitionInterface):
    def __init__(self, model_name):
        model = get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info

        self.model = None
        self.processor = None
        self.slice_bboxes_from_image = None
        self.batch_text_detection = None
        self.batch_recognition = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        from surya.model.recognition.model import load_model
        from surya.model.recognition.processor import load_processor
        from surya.input.processing import slice_bboxes_from_image
        from surya.recognition import batch_recognition

        self.model = load_model()
        self.processor = load_processor()
        self.slice_bboxes_from_image = slice_bboxes_from_image
        self.batch_recognition = batch_recognition

    def recognition(self, image, lang, lines, options=None):
        options = options or {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if isinstance(lines, dict):
            lines = TextDetectionResult(**lines)

        bboxes = [x.bbox for x in lines.bboxes]

        slices = self.slice_bboxes_from_image(image, bboxes)
        langs = [lang] * len(slices)

        rec_predictions, confidences = self.batch_recognition(slices, langs,
                                                    self.model, self.processor,
                                                    batch_size=options.get("batch_size"))

        text_lines = []
        for text, bbox, confidence in zip(rec_predictions, bboxes, confidences):
            text_lines.append(TextLine(
                text=text,
                bbox=bbox,
                confidence=confidence
            ))

        pred = TextRecognitionResult(
            text_lines=text_lines,
            languages=lang,
            image_bbox={"bbox": [0, 0, image.size[0], image.size[1]]}
        )

        return pred

