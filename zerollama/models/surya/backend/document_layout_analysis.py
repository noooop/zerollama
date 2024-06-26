from PIL import Image
from zerollama.tasks.ocr.document_layout_analysis.interface import DocumentLayoutAnalysisInterface
from zerollama.tasks.ocr.document_layout_analysis.protocol import DocumentLayoutAnalysisResult
from zerollama.tasks.ocr.document_layout_analysis.collection import get_model_by_name
from zerollama.tasks.ocr.text_line_detection.protocol import TextDetectionResult


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
        self.TextDetectionResult = None
        self.Bbox = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        from surya.settings import settings
        from surya.layout import batch_layout_detection
        from surya.model.detection.segformer import load_model, load_processor
        from surya.schema import TextDetectionResult, Bbox

        model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)

        self.labels = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", "Picture", "Figure",
                       "Section-header", "Table", "Text", "Title"]
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.model = model
        self.processor = processor
        self.batch_layout_detection = batch_layout_detection
        self.TextDetectionResult = TextDetectionResult
        self.Bbox = Bbox

    def detection(self, image, lines=None, options=None):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if lines is not None:
            if isinstance(lines, dict):
                lines = TextDetectionResult(**lines)
            polys = [{"polygon": self.Bbox(bbox=s.bbox).polygon, "confidence": s.confidence} for s in lines.bboxes]
            lines = self.TextDetectionResult(bboxes=polys,
                                             vertical_lines=[x.dict() for x in lines.vertical_lines],
                                             horizontal_lines=[x.dict() for x in lines.horizontal_lines],
                                             heatmap=None,
                                             affinity_map=None,
                                             image_bbox=[0, 0, image.size[0], image.size[1]])

        layout_predictions = self.batch_layout_detection([image], self.model, self.processor, [lines])

        bboxes = []
        for bbox in layout_predictions[0].bboxes:
            t = bbox.dict()
            t["id"] = self.label2id[bbox.label]
            bboxes.append(t)

        return DocumentLayoutAnalysisResult(bboxes=bboxes)
