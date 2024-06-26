
from zerollama.tasks.retriever.interface import Retriever


class BGERetriever(Retriever):
    family = "bge-retrieval"
    model_kwargs = {}
    header = ["name", "modelscope_name", "dimension", "sequence_length", "introduction"]
    info = [
        # name                           modelscope_name                 dimension    sequence_length    introduction
        ["BAAI/bge-m3",                  "Xorbits/bge-m3",               "1024",      "8192",            "multilingual; unified fine-tuning (dense, sparse, and colbert) from bge-m3-unsupervised"],
        ["BAAI/bge-m3-unsupervised",     "",                             "1024",      "8192",            "multilingual; contrastive learning from bge-m3-retromae"],
        ["BAAI/bge-m3-retromae",         "",                             "--",        "512",             "multilingual; extend the max_length of xlm-roberta to 8192 and further pretrained via retromae"],
        ["BAAI/bge-large-en-v1.5",       "Xorbits/bge-large-en-v1.5",    "1024",      "512",             "English model"],
        ["BAAI/bge-base-en-v1.5",        "Xorbits/bge-base-en-v1.5",     "768",       "512",             "English model"],
        ["BAAI/bge-small-en-v1.5",       "Xorbits/bge-small-en-v1.5",    "384",       "512",             "English model"],
    ]
    inference_backend = "zerollama.models.baai.backend.retriever:BGERetriever"


if __name__ == '__main__':
    def get_model(model_name):
        model_kwargs = {"local_files_only": False}

        model_class = BGERetriever.inference_backend
        module_name, class_name = model_class.split(":")
        import importlib

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = model_class(model_name=model_name, **model_kwargs)
        model.load()
        return model


    model_name = "BAAI/bge-m3"

    model = get_model(model_name)

    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    embeddings_1 = model.encode(sentences_1,
                                options={
                                    "batch_size": 12,
                                    "max_length": 8192,
                                }).vecs['dense_vecs']
    embeddings_2 = model.encode(sentences_2).vecs['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)