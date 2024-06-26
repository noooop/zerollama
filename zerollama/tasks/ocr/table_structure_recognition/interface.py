
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.ocr.table_structure_recognition.protocol import PROTOCOL
from zerollama.tasks.ocr.table_structure_recognition.protocol import TableStructureRecognitionModelConfig


class TableStructureRecognitionModel(ModelBase):
    protocol = PROTOCOL
    inference_backend = ""
    download_backend = ""

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return TableStructureRecognitionModelConfig(**model_config)


class TableStructureRecognitionInterface(object):
    protocol = PROTOCOL

    def load(self):
        raise NotImplementedError

    def recognition(self, image, options=None):
        raise NotImplementedError
