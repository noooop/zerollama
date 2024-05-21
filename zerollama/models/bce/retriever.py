
from zerollama.tasks.retriever.interface import Retriever

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BCERetriever(Retriever):
    family = "bce-embedding"
    model_kwargs = {}
    header = ["name", "hf_name", "modelscope_name"]
    info = [
        # name
        ["bce-embedding-base_v1", "maidalun1020/bce-embedding-base_v1", "maidalun/bce-embedding-base_v1"],

    ]
    inference_backend = "zerollama.models.bce.backend.retriever:BCERetriever"
    download_backend = "zerollama.models.bce.backend.download:download"


if __name__ == '__main__':
    def get_model(model_name):
        model_kwargs = {}

        model_class = BCERetriever.inference_backend
        module_name, class_name = model_class.split(":")
        import importlib

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = model_class(model_name=model_name, **model_kwargs)
        model.load()
        return model


    model_name = "bce-embedding-base_v1"

    model = get_model(model_name)

    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    embeddings_1 = model.encode(sentences_1).vecs['dense_vecs']
    embeddings_2 = model.encode(sentences_2).vecs['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)