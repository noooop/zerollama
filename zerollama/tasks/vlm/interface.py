
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.vlm.protocol import PROTOCOL
from zerollama.tasks.vlm.protocol import VLMModelConfig


class VLMModel(ModelBase):
    protocol = PROTOCOL
    inference_backend = ""
    download_backend = ""

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return VLMModelConfig(**model_config)


class VLMInterface(object):
    protocol = PROTOCOL

    def load(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def chat(self, messages, images, options=None):
        raise NotImplementedError
