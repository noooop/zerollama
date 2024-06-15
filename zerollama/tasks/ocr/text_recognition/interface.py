
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.ocr.text_recognition.protocol import PROTOCOL
from zerollama.tasks.ocr.text_recognition.protocol import TextRecognitionModelConfig


class TextRecognitionModel(ModelBase):
    protocol = PROTOCOL
    inference_backend = ""
    download_backend = ""

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return TextRecognitionModelConfig(**model_config)


class TextRecognitionInterface(object):
    protocol = PROTOCOL

    def load(self):
        raise NotImplementedError

    def recognition(self, image, lang, lines, options=None):
        raise NotImplementedError
