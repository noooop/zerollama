import torch
import requests
from zerollama.tasks.retriever.interface import RetrieverInterface
from zerollama.tasks.retriever.protocol import RetrieverResponse
from zerollama.tasks.retriever.collection import get_model_config_by_name, get_model_by_name
from zerollama.tasks.base.download import get_pretrained_model_name_or_path


class BGERetriever(RetrieverInterface):
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
        from FlagEmbedding import BGEM3FlagModel

        try:
            self.model = BGEM3FlagModel(self.pretrained_model_name_or_path,
                                        use_fp16=True,
                                        normalize_embeddings=True,
                                        device=self.device)
        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None

    @torch.no_grad()
    def encode(self, sentences, options=None):
        options = options or {}
        batch_size = options.get("batch_size", 12)
        max_length = options.get("max_length", 8192)
        return_dense = options.get("return_dense", True)
        return_sparse = options.get("return_sparse", False)
        return_colbert_vecs = options.get("return_colbert_vecs", False)

        kwargs = {"batch_size": batch_size,
                  "max_length": max_length,
                  "return_dense": return_dense,
                  "return_sparse": return_sparse,
                  "return_colbert_vecs": return_colbert_vecs}

        output = self.model.encode(sentences, **kwargs)
        return RetrieverResponse(model=self.model_name, vecs=output)

    @property
    def info(self):
        return self.model_info
