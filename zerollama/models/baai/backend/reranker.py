import torch
import requests
import numpy as np
from functools import partial
from zerollama.core.config.main import config_setup
from zerollama.tasks.reranker.interface import RerankerInterface
from zerollama.tasks.reranker.protocol import RerankerResponse
from zerollama.tasks.reranker.collection import get_model_config_by_name


class BGEReranker(RerankerInterface):
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

        self.model = None
        self.n_concurrent = 1

    def load(self):
        config_setup()

        if self.local_files_only:
            import huggingface_hub
            huggingface_hub.snapshot_download = partial(huggingface_hub.snapshot_download,
                                                        local_files_only=True)

        from FlagEmbedding import FlagReranker, FlagLLMReranker, LayerWiseFlagLLMReranker

        try:
            if self.model_info["type"] == "normal":
                self.model = FlagReranker(self.model_name,
                                          use_fp16=True,
                                          device=self.device)
            elif self.model_info["type"] == "LLM-based":
                self.model = FlagLLMReranker(self.model_name,
                                             use_fp16=True,
                                             device=self.device)
            elif self.model_info["type"] == "layerwise":
                self.model = LayerWiseFlagLLMReranker(self.model_name,
                                                      use_fp16=True,
                                                      device=self.device)
            else:
                raise ValueError(f"model type '{self.model_name}' not support")
        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None

    @torch.no_grad()
    def compute_score(self, sentence_pairs, options=None):
        options = options or {}

        batch_size = options.get("batch_size", 1)
        max_length = options.get("max_length", 1024)
        normalize = options.get("normalize", True)

        kwargs = {"batch_size": batch_size,
                  "max_length": max_length,
                  "normalize": normalize}

        if self.model_info["type"] == "layerwise":
            kwargs["cutoff_layers"] = options.get("cutoff_layers", [28])

        scores = self.model.compute_score(sentence_pairs, **kwargs)

        return RerankerResponse(model=self.model_name, vecs={"scores": np.array(scores)})

    @property
    def info(self):
        return self.model_info

