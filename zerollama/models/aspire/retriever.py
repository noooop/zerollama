
from zerollama.tasks.retriever.interface import Retriever


class ACGERetriever(Retriever):
    family = "acge"
    model_kwargs = {}
    header = ["name", "modelscope_name", "matryoshka"]
    info = [
        # name                          modelscope_name                 matryoshka
        ["aspire/acge_text_embedding", "yangjhchs/acge_text_embedding", True],
    ]
    inference_backend = "zerollama.microservices.inference.sentence_transformer_green.retriever:SentenceTransformerRetriever"


if __name__ == '__main__':
    def get_model(model_name):
        model_kwargs = {"local_files_only": False}

        model_class = ACGERetriever.inference_backend
        module_name, class_name = model_class.split(":")
        import importlib

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = model_class(model_name=model_name, **model_kwargs)
        model.load()
        return model


    model_name = "aspire/acge_text_embedding"

    model = get_model(model_name)

    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    matryoshka_dim = 512

    embeddings_1 = model.encode(sentences_1, options={"matryoshka_dim": matryoshka_dim}).vecs['dense_vecs']
    embeddings_2 = model.encode(sentences_2, options={"matryoshka_dim": matryoshka_dim}).vecs['dense_vecs']
    print(embeddings_1.shape, embeddings_2.shape)
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)