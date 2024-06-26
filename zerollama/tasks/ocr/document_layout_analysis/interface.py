
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.ocr.document_layout_analysis.protocol import PROTOCOL
from zerollama.tasks.ocr.document_layout_analysis.protocol import DocumentLayoutAnalysisModelConfig


class DocumentLayoutAnalysisModel(ModelBase):
    protocol = PROTOCOL
    inference_backend = ""
    download_backend = ""

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return DocumentLayoutAnalysisModelConfig(**model_config)


class DocumentLayoutAnalysisInterface(object):
    protocol = PROTOCOL

    def load(self):
        raise NotImplementedError

    def detection(self, image, lines=None, options=None):
        raise NotImplementedError
