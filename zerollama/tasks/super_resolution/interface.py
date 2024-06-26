
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.super_resolution.protocol import PROTOCOL
from zerollama.tasks.super_resolution.protocol import SRModelConfig


class SuperResolutionModel(ModelBase):
    protocol = PROTOCOL
    download_backend = ""

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return SRModelConfig(**model_config)


class SuperResolutionInterface(object):
    protocol = PROTOCOL

    def load(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def sr(self, img_lr, **options):
        raise NotImplementedError


