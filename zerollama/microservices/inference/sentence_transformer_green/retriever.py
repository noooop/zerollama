
import torch
import requests
from sklearn.preprocessing import normalize
from zerollama.tasks.retriever.interface import RetrieverInterface
from zerollama.tasks.retriever.protocol import RetrieverResponse
from zerollama.tasks.retriever.collection import get_model_by_name
from zerollama.tasks.base.download import get_pretrained_model_name_or_path


class SentenceTransformerRetriever(RetrieverInterface):
    def __init__(self, model_name, local_files_only=True, device="cuda"):
        model = get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

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
                                                                               get_model_by_name=get_model_by_name)
        self.matryoshka_support = self.model_info.get("matryoshka", False)

        self.model = None
        self.n_concurrent = 1

    def load(self):
        from sentence_transformers import SentenceTransformer

        try:
            self.model = SentenceTransformer(str(self.pretrained_model_name_or_path), device=self.device)
        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None

    @torch.no_grad()
    def encode(self, sentences, options=None):
        options = options or {}
        normalize_embeddings = options.get("normalize_embeddings", True)

        if self.matryoshka_support and "matryoshka_dim" in options:
            matryoshka_dim = options["matryoshka_dim"]
            dense_vecs = self.model.encode(sentences)
            dense_vecs = dense_vecs[..., :matryoshka_dim]
            dense_vecs = normalize(dense_vecs, norm="l2", axis=1)
        else:
            kwargs = {"normalize_embeddings": normalize_embeddings}
            dense_vecs = self.model.encode(sentences, **kwargs)

        return RetrieverResponse(model=self.model_name, vecs={"dense_vecs": dense_vecs})

    @property
    def info(self):
        return self.model_info