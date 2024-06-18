# Adapted from
# https://github.com/infiniflow/ragflow/blob/main/deepdoc/vision/ocr.py

import cv2
import os
from pathlib import Path
import copy
import numpy as np
from PIL import Image
from zerollama.tasks.ocr.text_recognition.interface import TextRecognitionInterface
from zerollama.tasks.ocr.text_recognition.protocol import TextLine, TextRecognitionResult
from zerollama.tasks.ocr.text_line_detection.protocol import TextDetectionResult
from zerollama.tasks.ocr.text_recognition.collection import get_model_by_name

ragflow_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent / "ragflow"


class DeepDocTextRecognition(TextRecognitionInterface):
    def __init__(self, model_name):
        model = get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info

        self.model = None
        self.drop_score = 0.5
        self.crop_image_res_index = 0
        self.sorted_boxes = None
        self.get_rotate_crop_image = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        import sys
        sys.path.append(str(ragflow_path))
        from deepdoc.vision.ocr import TextRecognizer, get_project_base_directory
        from zerollama.tasks.ocr.utils import sorted_boxes, get_rotate_crop_image

        model_dir = os.path.join(
            get_project_base_directory(),
            "rag/res/deepdoc")

        self.model = TextRecognizer(model_dir)
        self.sorted_boxes = sorted_boxes
        self.get_rotate_crop_image = get_rotate_crop_image

    def recognition(self, image, lang, lines, options=None):
        options = options or {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        ori_im = np.array(image)

        if isinstance(lines, dict):
            lines = TextDetectionResult(**lines)

        dt_boxes = np.array([x.polygon for x in lines.bboxes])
        dt_boxes = self.sorted_boxes(dt_boxes)

        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, elapse = self.model(img_crop_list)

        text_lines = []
        for i, rec_result in enumerate(rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                text_lines.append(TextLine(
                    text=text,
                    bbox=[dt_boxes[i][0, 0], dt_boxes[i][0, 1], dt_boxes[i][1, 0], dt_boxes[i][2, 1]],
                    confidence=score
                ))

        pred = TextRecognitionResult(
            text_lines=text_lines,
            languages=lang,
            image_bbox={"bbox": [0, 0, image.size[0], image.size[1]]}
        )

        return pred

