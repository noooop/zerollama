import torch
import requests
import numpy as np
from zerollama.tasks.reranker.interface import RerankerInterface
from zerollama.tasks.reranker.protocol import RerankerResponse
from zerollama.tasks.reranker.collection import get_model_config_by_name, get_model_by_name
from zerollama.tasks.base.download import get_pretrained_model_name_or_path


class BCEReranker(RerankerInterface):
    def __init__(self, model_name, local_files_only=True, device="cuda"):
        model_config = get_model_config_by_name(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.local_files_only = local_files_only
        self.trust_remote_code = self.model_config.model_kwargs.get("trust_remote_code", False)
        self.pretrained_model_name_or_path = get_pretrained_model_name_or_path(model_name=model_name,
                                                                               local_files_only=local_files_only,
                                                                               get_model_by_name=get_model_by_name,
                                                                               get_model_config_by_name=get_model_config_by_name)

        self.model = None
        self.n_concurrent = 1

    def load(self):
        from BCEmbedding import RerankerModel

        try:
            self.model = RerankerModel(self.pretrained_model_name_or_path, device=self.device)
        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None

    @torch.no_grad()
    def compute_score(self, sentence_pairs, options=None):
        options = options or {}
        scores = self.model.compute_score(sentence_pairs)

        return RerankerResponse(model=self.model_name, vecs={"scores": np.array(scores)})

    @property
    def info(self):
        return self.model_info

