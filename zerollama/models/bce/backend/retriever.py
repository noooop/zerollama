import torch
import requests
from functools import partial
from zerollama.core.config.main import config_setup
from zerollama.tasks.retriever.interface import RetrieverInterface
from zerollama.tasks.retriever.protocol import RetrieverResponse
from zerollama.tasks.retriever.collection import get_model_config_by_name


class BCERetriever(RetrieverInterface):
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

        from pathlib import Path
        from BCEmbedding import EmbeddingModel

        def get_cache_dir():
            import os
            default_cache_dir = Path.home().joinpath('.cache', 'modelscope')
            base_path = os.getenv('MODELSCOPE_CACHE', os.path.join(default_cache_dir, 'hub'))
            return base_path

        cache_dir = get_cache_dir()

        model_name_or_path = Path(cache_dir) / self.model_info["modelscope_name"]

        try:
            self.model = EmbeddingModel(model_name_or_path, device=self.device)
        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None

    @torch.no_grad()
    def encode(self, sentences, options=None):
        dense_vecs = self.model.encode(sentences)
        return RetrieverResponse(model=self.model_name, vecs={"dense_vecs": dense_vecs})

    @property
    def info(self):
        return self.model_info

