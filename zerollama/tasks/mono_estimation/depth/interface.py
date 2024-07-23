
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.mono_estimation.depth.protocol import PROTOCOL
from zerollama.tasks.mono_estimation.depth.protocol import DepthEstimationModelConfig


class DepthEstimationModel(ModelBase):
    protocol = PROTOCOL
    inference_backend = ""
    download_backend = ""

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return DepthEstimationModelConfig(**model_config)


class DepthEstimationInterface(object):
    protocol = PROTOCOL

    def load(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def generate(self, prompt, negative_prompt=None, options=None):
        raise NotImplementedError
