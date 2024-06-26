
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.ocr.reading_order_detection.protocol import PROTOCOL
from zerollama.tasks.ocr.reading_order_detection.protocol import ReadingOrderDetectionModelConfig


class ReadingOrderDetectionModel(ModelBase):
    protocol = PROTOCOL
    inference_backend = ""
    download_backend = ""

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return ReadingOrderDetectionModelConfig(**model_config)


class ReadingOrderDetectionInterface(object):
    protocol = PROTOCOL

    def load(self):
        raise NotImplementedError

    def detection(self, image, layout, options=None):
        raise NotImplementedError
