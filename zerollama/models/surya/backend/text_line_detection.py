
from PIL import Image
from zerollama.tasks.ocr.text_line_detection.interface import TextLineDetectionInterface
from zerollama.tasks.ocr.text_line_detection.protocol import TextDetectionResult
from zerollama.tasks.ocr.text_line_detection.collection import get_model_by_name


class SuryaTextLineDetection(TextLineDetectionInterface):
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
        self.batch_text_detection = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        from surya.detection import batch_text_detection
        from surya.model.detection.segformer import load_model, load_processor

        model, processor = load_model(), load_processor()

        self.model = model
        self.processor = processor
        self.batch_text_detection = batch_text_detection

    def detection(self, image, options=None):
        options = options or {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        predictions = self.batch_text_detection([image], self.model, self.processor, **options)

        bboxes = []
        for x in predictions[0].bboxes:
            t = {"bbox": x.bbox,
                 "confidence": x.confidence}
            bboxes.append(t)

        return TextDetectionResult(bboxes=bboxes)

    def detection_raw(self, image, options=None):
        options = options or {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        predictions = self.batch_text_detection([image], self.model, self.processor, **options)
        return predictions
