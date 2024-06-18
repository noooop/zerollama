import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import numpy as np
from pathlib import Path
from PIL import Image
from easydict import EasyDict as edict
from zerollama.tasks.ocr.text_recognition.interface import TextRecognitionInterface
from zerollama.tasks.ocr.text_recognition.protocol import TextLine, TextRecognitionResult
from zerollama.tasks.ocr.text_line_detection.protocol import TextDetectionResult
from zerollama.tasks.ocr.text_recognition.collection import get_model_by_name


class PaddleOCRTextRecognition(TextRecognitionInterface):
    def __init__(self, model_name, lang=None, **kwargs):
        model_class = get_model_by_name(model_name)
        model_config = model_class.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.lang = lang or model_class.lang
        self.kwargs = kwargs

        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info

        self.model = None
        self.params = None
        self.sorted_boxes = None
        self.get_rotate_crop_image = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        import paddleocr
        from paddleocr.tools.infer.predict_rec import TextRecognizer
        from paddleocr.paddleocr import get_model_config, confirm_model_dir_url, BASE_DIR, maybe_download
        from zerollama.tasks.ocr.utils import sorted_boxes, get_rotate_crop_image
        import zerollama.models.paddleocr.utils as utils

        self.params = utils.engine_args | utils.text_recognizer_args | utils.multi_process_args | self.model_info
        self.params.update(**self.kwargs)
        self.params = edict(self.params)

        if self.params.ocr_version in ['PP-OCRv3', 'PP-OCRv4']:
            self.params.rec_image_shape = "3, 48, 320"
        else:
            self.params.rec_image_shape = "3, 32, 320"

        rec_model_config = get_model_config('OCR', self.params.ocr_version, 'rec', self.lang)
        if self.params.rec_char_dict_path is None:
            self.params.rec_char_dict_path = str(Path(paddleocr.paddleocr.__file__).parent / rec_model_config['dict_path'])

        self.params.rec_model_dir, det_url = confirm_model_dir_url(
            None,
            os.path.join(BASE_DIR, 'whl', 'rec', self.lang),
            rec_model_config['url'])

        maybe_download(self.params.rec_model_dir, det_url)

        self.model = TextRecognizer(self.params)
        self.sorted_boxes = sorted_boxes
        self.get_rotate_crop_image = get_rotate_crop_image

    def recognition(self, image, lang, lines, options=None):
        options = options or {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if isinstance(lines, dict):
            lines = TextDetectionResult(**lines)

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

