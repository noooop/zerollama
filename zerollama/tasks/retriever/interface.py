
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.retriever.protocol import PROTOCOL
from zerollama.tasks.retriever.protocol import RetrieverResponse, RetrieverModelConfig


class Retriever(ModelBase):
    protocol = PROTOCOL
    download_backend = "zerollama.inference_backend.transformers_green.download:download"

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return RetrieverModelConfig(**model_config)


class RetrieverInterface(object):
    protocol = PROTOCOL

    def load(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def encode(self, sentences, **options) -> RetrieverResponse:
        """

        :param sentences:
        :param options:
        :return:
        """
        raise NotImplementedError

