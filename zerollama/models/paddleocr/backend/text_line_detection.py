
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
from easydict import EasyDict as edict
from zerollama.tasks.ocr.text_line_detection.interface import TextLineDetectionInterface
from zerollama.tasks.ocr.text_line_detection.protocol import TextDetectionResult
from zerollama.tasks.ocr.text_line_detection.collection import get_model_by_name


class PaddleOCRTextLineDetection(TextLineDetectionInterface):
    def __init__(self, model_name, lang=None, **kwargs):
        model_class = get_model_by_name(model_name)
        model_config = model_class.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.lang = lang or model_class.lang
        self.kwargs = kwargs

        self.model_class = model_class
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info

        self.model = None
        self.params = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        from paddleocr.tools.infer.predict_det import TextDetector
        from paddleocr.paddleocr import get_model_config, confirm_model_dir_url, BASE_DIR, maybe_download
        import zerollama.models.paddleocr.utils as utils

        self.params = utils.engine_args | utils.text_detector_args | utils.multi_process_args | self.model_class.DB_parmas | self.model_info
        self.params.update(**self.kwargs)
        self.params = edict(self.params)

        det_model_config = get_model_config('OCR', self.params.ocr_version, 'det', self.lang)
        self.params.det_model_dir, det_url = confirm_model_dir_url(
            None,
            os.path.join(BASE_DIR, 'whl', 'det', self.lang),
            det_model_config['url'])

        maybe_download(self.params.det_model_dir, det_url)

        self.model = TextDetector(self.params)

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
