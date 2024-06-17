
import os
from pathlib import Path
import numpy as np
from PIL import Image
from zerollama.tasks.ocr.table_structure_recognition.interface import TableStructureRecognitionInterface
from zerollama.tasks.ocr.table_structure_recognition.protocol import TableStructureRecognitionResult
from zerollama.tasks.ocr.table_structure_recognition.collection import get_model_by_name

ragflow_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent / "ragflow"


class DeepDocTableStructureRecognition(TableStructureRecognitionInterface):
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

        from deepdoc.vision.table_structure_recognizer import TableStructureRecognizer

        self.labels = TableStructureRecognizer.labels
        self.label2id = {l.lower(): i for i, l in enumerate(self.labels)}
        self.model = TableStructureRecognizer()

    def recognition(self, image, options=None):
        options = options or {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = np.array(image)

        predictions = self.model([image], **options)

        bboxes = []
        for bbox in predictions[0]:
            t = {"bbox": [bbox["x0"], bbox["top"], bbox["x1"], bbox["bottom"]],
                 "label": bbox["label"],
                 "id": self.label2id[bbox["label"].lower()],
                 "confidence": bbox["score"]}
            bboxes.append(t)

        return TableStructureRecognitionResult(bboxes=bboxes)

