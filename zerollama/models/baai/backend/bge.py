import torch
import requests
from zerollama.core.config.main import config_setup
from zerollama.tasks.retriever.interface import RetrieverInterface
from functools import partial


class BGEM3(RetrieverInterface):
    def __init__(self, model_name, local_files_only=True, device="cuda"):
        self.model = None

        self.device = device
        self.model_name = model_name
        self.local_files_only = local_files_only

    def load(self):
        config = config_setup()

        if self.local_files_only:
            import huggingface_hub
            huggingface_hub.snapshot_download = partial(huggingface_hub.snapshot_download,
                                                        local_files_only=True)

        from FlagEmbedding import BGEM3FlagModel

        try:
            self.model = BGEM3FlagModel(self.model_name,
                                        use_fp16=True,
                                        normalize_embeddings=True,
                                        device=self.device)
        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None

    @torch.no_grad()
    def encode(self, sentences, **options):
        batch_size = options.get("batch_size", 12)
        max_length = options.get("max_length", 8192)
        return_dense = options.get("return_dense", True)
        return_sparse = options.get("return_sparse", False)
        return_colbert_vecs = options.get("return_colbert_vecs", False)

        kwargs = {"batch_size": batch_size,
                  "max_length": max_length,
                  "return_dense": return_dense,
                  "return_sparse": return_sparse,
                  "return_colbert_vecs": return_colbert_vecs,}

        output = self.model.encode(sentences, **kwargs)
        return output
