
import os
from pathlib import Path

import numpy as np
from PIL import Image
from zerollama.tasks.ocr.text_line_detection.interface import TextLineDetectionInterface
from zerollama.tasks.ocr.text_line_detection.protocol import TextDetectionResult
from zerollama.tasks.ocr.text_line_detection.collection import get_model_by_name

ragflow_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent / "ragflow"


class DeepDocTextLineDetection(TextLineDetectionInterface):
    def __init__(self, model_name):
        model = get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info

        self.model = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        import sys
        sys.path.append(str(ragflow_path))
        from deepdoc.vision.ocr import TextDetector, get_project_base_directory

        model_dir = os.path.join(
            get_project_base_directory(),
            "rag/res/deepdoc")

        self.model = TextDetector(model_dir)

    def detection(self, image, options=None):
        options = options or {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = np.array(image)

        predictions = self.model(image)
        dt_boxes = predictions[0]

        bboxes = []
        for i in range(dt_boxes.shape[0]):
            t = {"bbox": [dt_boxes[i, 0, 0], dt_boxes[i, 0, 1], dt_boxes[i, 1, 0], dt_boxes[i, 2, 1]],
                 "confidence": 1.0}
            bboxes.append(t)

        return TextDetectionResult(bboxes=bboxes, vertical_lines=[], horizontal_lines=[])
