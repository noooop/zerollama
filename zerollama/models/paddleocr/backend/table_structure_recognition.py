import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
from pathlib import Path
from easydict import EasyDict as edict
from zerollama.tasks.ocr.table_structure_recognition.interface import TableStructureRecognitionInterface
from zerollama.tasks.ocr.table_structure_recognition.protocol import TableStructureRecognitionResult
from zerollama.tasks.ocr.table_structure_recognition.collection import get_model_by_name


class PaddleOCRTableStructureRecognition(TableStructureRecognitionInterface):
    def __init__(self, model_name, lang=None, **kwargs):
        model_class = get_model_by_name(model_name)
        model_config = model_class.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.kwargs = kwargs

        self.model_class = model_class
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.lang = lang or self.model_config.info["lang"]

        self.model = None
        self.params = None
        self.labels = None
        self.label2id = None
        self.n_concurrent = 1

    def load(self):
        import paddleocr
        from paddleocr.ppstructure.table.predict_structure import TableStructurer
        from paddleocr.paddleocr import get_model_config, confirm_model_dir_url, BASE_DIR, maybe_download
        import zerollama.models.paddleocr.utils as utils

        self.params = utils.engine_args | utils.tabel_args | utils.multi_process_args | self.model_info
        self.params.update(**self.kwargs)
        self.params = edict(self.params)

        if self.params.structure_version == 'PP-Structure':
            self.params.merge_no_span_structure = False

        table_model_config = get_model_config('STRUCTURE', self.params.structure_version, 'table', self.lang)

        if self.params.table_char_dict_path is None:
            self.params.table_char_dict_path = str(Path(paddleocr.paddleocr.__file__).parent / table_model_config['dict_path'])

        self.params.table_model_dir, table_url = confirm_model_dir_url(
            None,
            os.path.join(BASE_DIR, 'whl', 'table', self.lang),
            table_model_config['url'])

        maybe_download(self.params.table_model_dir, table_url)

        self.model = TableStructurer(self.params)

    def recognition(self, image, options=None):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = np.array(image)

        (structure_str_list, bbox_list), elapse = self.model(image)

        bboxes = []
        for bbox in bbox_list:
            t = {"bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                 "label": "",
                 "id": 0,
                 "confidence": 1.0}
            bboxes.append(t)

        return TableStructureRecognitionResult(bboxes=bboxes)

