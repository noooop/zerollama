
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.text2image.protocol import PROTOCOL
from zerollama.tasks.text2image.protocol import Text2ImageModelConfig


class Text2ImageModel(ModelBase):
    protocol = PROTOCOL
    inference_backend = ""
    download_backend = ""

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return Text2ImageModelConfig(**model_config)


class Text2ImageInterface(object):
    protocol = PROTOCOL

    def load(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def generate(self, prompt, negative_prompt=None, options=None):
        raise NotImplementedError
