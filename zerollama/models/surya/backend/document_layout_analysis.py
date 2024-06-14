
from PIL import Image
from zerollama.tasks.ocr.document_layout_analysis.interface import DocumentLayoutAnalysisInterface
from zerollama.tasks.ocr.document_layout_analysis.protocol import DocumentLayoutAnalysisResult
from zerollama.tasks.ocr.document_layout_analysis.collection import get_model_by_name


class SuryaDocumentLayoutAnalysis(DocumentLayoutAnalysisInterface):
    def __init__(self, model_name):
        model = get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info

        self.model = None
        self.labels = None
        self.label2id = None
        self.processor = None
        self.batch_layout_detection = None
        self.text_line_detection = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        from surya.settings import settings
        from surya.layout import batch_layout_detection
        from surya.model.detection.segformer import load_model, load_processor
        from zerollama.models.surya.backend.text_line_detection import SuryaTextLineDetection

        model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)

        self.labels = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", "Picture", "Figure", "Section-header", "Table", "Text", "Title"]
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.model = model
        self.processor = processor
        self.batch_layout_detection = batch_layout_detection
        self.text_line_detection = SuryaTextLineDetection("surya_tld")
        self.text_line_detection.load()

    def detection(self, image, options=None):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        line_predictions = self.text_line_detection.detection_raw(image)
        layout_predictions = self.batch_layout_detection([image], self.model, self.processor, line_predictions)

        bboxes = []
        for bbox in layout_predictions[0].bboxes:
            t = bbox.dict()
            t["id"] = self.label2id[bbox.label]
            bboxes.append(t)

        return DocumentLayoutAnalysisResult(bboxes=bboxes)
