
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.reranker.protocol import PROTOCOL
from zerollama.tasks.reranker.protocol import RerankerResponse, RerankerModelConfig


class Reranker(ModelBase):
    protocol = PROTOCOL
    download_backend = "zerollama.microservices.inference.transformers_green.download:download"

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return RerankerModelConfig(**model_config)


class RerankerInterface(object):
    protocol = PROTOCOL

    def load(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def compute_score(self, sentence_pairs, options=None) -> RerankerResponse:
        """

        :param sentence_pairs:
        :param options:
        :return:
        """
        raise NotImplementedError

