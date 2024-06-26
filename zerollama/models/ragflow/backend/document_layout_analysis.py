import os
from pathlib import Path

import numpy as np
from PIL import Image
from zerollama.tasks.ocr.document_layout_analysis.interface import DocumentLayoutAnalysisInterface
from zerollama.tasks.ocr.document_layout_analysis.protocol import DocumentLayoutAnalysisResult
from zerollama.tasks.ocr.document_layout_analysis.collection import get_model_by_name
from zerollama.tasks.ocr.text_line_detection.protocol import TextDetectionResult

ragflow_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent / "ragflow"


class DeepDocDocumentLayoutAnalysis(DocumentLayoutAnalysisInterface):
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
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        import sys
        sys.path.append(str(ragflow_path))
        from deepdoc.vision.recognizer import Recognizer
        from deepdoc.vision.layout_recognizer import LayoutRecognizer, get_project_base_directory

        model_dir = os.path.join(
            get_project_base_directory(),
            "rag/res/deepdoc")

        self.labels = LayoutRecognizer.labels
        self.label2id = {l.lower(): i for i, l in enumerate(self.labels)}
        self.model = Recognizer(self.labels, self.model_name.split("/")[-1], model_dir)

    def detection(self, image, lines=None, options=None):
        options = options or {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = np.array(image)

        predictions = self.model([image], **options)

        bboxes = []
        for bbox in predictions[0]:
            t = {"bbox": bbox["bbox"],
                 "label": bbox["type"],
                 "id": self.label2id[bbox["type"].lower()],
                 "confidence": bbox["score"]}
            bboxes.append(t)

        return DocumentLayoutAnalysisResult(bboxes=bboxes)
