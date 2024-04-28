

from zerollama.tasks.retriever.interface import Retriever


class BGEM3(Retriever):
    family = "BGE"
    model_kwargs = {}
    header = ["name", "dimension", "sequence_length", "introduction"]
    info = [
        # name                           dimension    sequence_length    introduction
        ["BAAI/bge-m3",                  "1024",      "8192",            "multilingual; unified fine-tuning (dense, sparse, and colbert) from bge-m3-unsupervised"],
        ["BAAI/bge-m3-unsupervised",     "1024",      "8192",            "multilingual; contrastive learning from bge-m3-retromae"],
        ["BAAI/bge-m3-retromae",         "--",        "512",             "multilingual; extend the max_length of xlm-roberta to 8192 and further pretrained via retromae"],
        ["BAAI/bge-large-en-v1.5",       "1024",      "512",             "English model"],
        ["BAAI/bge-base-en-v1.5",        "768",       "512",             "English model"],
        ["BAAI/bge-small-en-v1.5",       "384",       "512",             "English model"],
    ]
    inference_backend = "zerollama.models.baai.bge_inference_backend:BGEM3"


def get_model(model_name):
    model_kwargs = {}

    model_class = BGEM3.inference_backend
    module_name, class_name = model_class.split(":")
    import importlib

    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    model = model_class(model_name=model_name, **model_kwargs)
    model.load()
    return model


if __name__ == '__main__':
    model_name = "BAAI/bge-m3"

    model = get_model(model_name)

    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    embeddings_1 = model.encode(sentences_1,
                                batch_size=12,
                                max_length=8192,
                                # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                )['dense_vecs']
    embeddings_2 = model.encode(sentences_2)['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)