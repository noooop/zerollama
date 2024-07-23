import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
from pathlib import Path
from easydict import EasyDict as edict
from zerollama.tasks.ocr.document_layout_analysis.interface import DocumentLayoutAnalysisInterface
from zerollama.tasks.ocr.document_layout_analysis.protocol import DocumentLayoutAnalysisResult
from zerollama.tasks.ocr.document_layout_analysis.collection import get_model_by_name


class PaddleOCRDocumentLayoutAnalysis(DocumentLayoutAnalysisInterface):
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
        self.labels = None
        self.label2id = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup
        config_setup()

        import paddleocr
        from paddleocr.ppstructure.layout.predict_layout import LayoutPredictor
        from paddleocr.paddleocr import get_model_config, confirm_model_dir_url, BASE_DIR, maybe_download
        import zerollama.models.paddleocr.utils as utils

        self.params = utils.engine_args | utils.layout_args | utils.multi_process_args | self.model_info
        self.params.update(**self.kwargs)
        self.params = edict(self.params)

        layout_model_config = get_model_config('STRUCTURE', self.params.structure_version, 'layout', self.lang)

        if self.params.layout_dict_path is None:
            self.params.layout_dict_path = str(Path(paddleocr.paddleocr.__file__).parent / layout_model_config['dict_path'])

        self.params.layout_model_dir, layout_url = confirm_model_dir_url(
            None,
            os.path.join(BASE_DIR, 'whl', 'layout', self.lang),
            layout_model_config['url'])

        maybe_download(self.params.layout_model_dir, layout_url)

        self.model = LayoutPredictor(self.params)
        self.labels = self.model.postprocess_op.labels
        self.label2id = {l.lower(): i for i, l in enumerate(self.labels)}

    def detection(self, image, lines=None, options=None):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = np.array(image)

        predictions = self.model(image)

        bboxes = []
        for bbox in predictions[0]:
            t = {"bbox": bbox["bbox"],
                 "label": bbox["label"],
                 "id": self.label2id[bbox["label"].lower()],
                 "confidence": 1.0}
            bboxes.append(t)

        return DocumentLayoutAnalysisResult(bboxes=bboxes, class_names=self.labels)
