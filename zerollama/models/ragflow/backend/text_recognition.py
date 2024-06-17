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
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        import sys
        sys.path.append(str(ragflow_path))
        from deepdoc.vision.ocr import TextRecognizer, get_project_base_directory

        model_dir = os.path.join(
            get_project_base_directory(),
            "rag/res/deepdoc")

        self.model = TextRecognizer(model_dir)

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(np.float32(points), pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

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

