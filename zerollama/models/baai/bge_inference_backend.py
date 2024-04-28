import torch
from zerollama.core.config.main import config_setup
from zerollama.tasks.retriever.interface import RetrieverInterface


class BGEM3(RetrieverInterface):
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.model_name = model_name
        self.model = None

    def load(self):
        config = config_setup()

        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(self.model_name,
                                    use_fp16=True,
                                    normalize_embeddings=True,
                                    device=self.device)

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
