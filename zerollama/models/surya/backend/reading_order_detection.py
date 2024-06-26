from PIL import Image
from zerollama.tasks.ocr.reading_order_detection.interface import ReadingOrderDetectionInterface
from zerollama.tasks.ocr.reading_order_detection.protocol import ReadingOrderDetectionResult
from zerollama.tasks.ocr.document_layout_analysis.protocol import DocumentLayoutAnalysisResult
from zerollama.tasks.ocr.reading_order_detection.collection import get_model_by_name


class SuryaReadingOrderDetection(ReadingOrderDetectionInterface):
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
        self.batch_ordering = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        from surya.ordering import batch_ordering
        from surya.model.ordering.processor import load_processor
        from surya.model.ordering.model import load_model

        self.model = load_model()
        self.processor = load_processor()
        self.batch_ordering = batch_ordering

    def detection(self, image, layout, options=None):
        options = options or {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if isinstance(layout, dict):
            layout = DocumentLayoutAnalysisResult(**layout)

        bboxes = [x.bbox for x in layout.bboxes]

        order_predictions = self.batch_ordering([image], [bboxes], self.model, self.processor, **options)
        return ReadingOrderDetectionResult(bboxes=[{"bbox": x.bbox, "position": x.position}
                                                   for x in order_predictions[0].bboxes])

