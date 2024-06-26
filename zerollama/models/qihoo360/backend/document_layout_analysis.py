
import os
import PIL.Image
from pathlib import Path
from zerollama.tasks.ocr.document_layout_analysis.interface import DocumentLayoutAnalysisInterface
from zerollama.tasks.ocr.document_layout_analysis.collection import get_model_by_name
from zerollama.tasks.ocr.document_layout_analysis.protocol import DocumentLayoutAnalysisResult

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
weight_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent / "models/360LayoutAnalysis"


class LayoutAnalysis360(DocumentLayoutAnalysisInterface):

    def __init__(self, model_name):
        model = get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info

        self.model = None
        self.weight_dtype = None
        self.n_concurrent = 1

    def load(self):
        from ultralytics import YOLO
        self.model = YOLO(str(weight_path / self.model_name.split("/")[-1]))

    def detection(self, image, lines=None, options=None):
        options = options or {}

        bboxes = []
        for bbox in self.model(source=PIL.Image.fromarray(image), **options)[0].summary():
            t = {"id": bbox["class"],
                 "label": bbox["name"],
                 "bbox": [bbox["box"]["x1"], bbox["box"]["y1"],bbox["box"]["x2"], bbox["box"]["y2"]],
                 "confidence": bbox["confidence"]}
            bboxes.append(t)

        return DocumentLayoutAnalysisResult(bboxes=bboxes)



