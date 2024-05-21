
from zerollama.tasks.retriever.interface import Retriever


class M3ERetriever(Retriever):
    family = "m3e"
    model_kwargs = {}
    header = ["name", "size", "dimension", "zh", "en", "s2s", "s2p", "s2c", "open source", "compatibility"]
    info = [
        # name                参数数量  维度   中文   英文  s2s   s2p  s2c   开源  兼容性
        ["moka-ai/m3e-small", "24M",  "512", "是", "否", "是", "否", "否", "是", "优"],
        ["moka-ai/m3e-base",  "110M", "768", "是", "是", "是", "是", "否", "是", "优"],
        ["moka-ai/m3e-large", "340M", "768", "是", "否", "是", "是", "否", "是", "优"],
    ]
    inference_backend = "zerollama.microservices.inference.sentence_transformer_green.retriever:SentenceTransformerRetriever"


if __name__ == '__main__':
    def get_model(model_name):
        model_kwargs = {"local_files_only": False}

        model_class = M3ERetriever.inference_backend
        module_name, class_name = model_class.split(":")
        import importlib

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = model_class(model_name=model_name, **model_kwargs)
        model.load()
        return model


    model_name = "moka-ai/m3e-small"

    model = get_model(model_name)

    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    embeddings_1 = model.encode(sentences_1).vecs['dense_vecs']
    embeddings_2 = model.encode(sentences_2).vecs['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)