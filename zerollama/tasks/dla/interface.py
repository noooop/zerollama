
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.dla.protocol import PROTOCOL
from zerollama.tasks.dla.protocol import DLAModelConfig


class DLAModel(ModelBase):
    protocol = PROTOCOL
    inference_backend = ""
    download_backend = ""

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return DLAModelConfig(**model_config)


class DLAInterface(object):
    protocol = PROTOCOL

    def load(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def chat(self, messages, images, options=None):
        raise NotImplementedError
