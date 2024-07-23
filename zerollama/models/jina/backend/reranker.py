import torch
import requests
import numpy as np
from zerollama.tasks.reranker.interface import RerankerInterface
from zerollama.tasks.reranker.protocol import RerankerResponse
from zerollama.tasks.reranker.collection import get_model_config_by_name


class JinaReranker(RerankerInterface):
    def __init__(self, model_name, local_files_only=True, device="cuda"):
        model_config = get_model_config_by_name(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.local_files_only = local_files_only
        self.model = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup

        config_setup()

        from transformers import AutoModelForSequenceClassification

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                local_files_only=self.local_files_only
            )
            self.model.to(self.device)
            self.model.eval()
        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None

    @torch.no_grad()
    def compute_score(self, sentence_pairs, options=None):
        options = options or {}

        max_length = options.get("max_length", 1024)
        kwargs = {"max_length": max_length}

        scores = self.model.compute_score(sentence_pairs, **kwargs)

        return RerankerResponse(model=self.model_name, vecs={"scores": np.array(scores)})

    @property
    def info(self):
        return self.model_info
