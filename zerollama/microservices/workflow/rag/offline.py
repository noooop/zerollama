import pickle
import numpy as np
from tqdm import trange
from hashlib import md5
from zerollama.scaffold.documents.splitter.main import text_window_parser
from zerollama.tasks.retriever.engine.client import RetrieverClient
from zerollama.core.config.main import config_setup


def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def text2vec(collection, embedding_model, batchsize=128):
    config = config_setup()
    file = list((config.rag.path / collection).glob("*.txt"))[0]
    pickle_name = md5(f"zerollama:{collection}:{embedding_model}:embeddings".encode("utf-8")).hexdigest()

    if (file.parent / (pickle_name + '.pkl')).exists():
        print(collection, embedding_model, "exists")
        return

    book_name = file.stem.split("-")[0]
    lines, sentence_list, nodes = text_window_parser(file)
    print(book_name)

    print(f"一共有{len(lines)}个段落")
    print(f"一共有{len(sentence_list)}个句子, 最大长度为{max(len(line['text']) for line in sentence_list)}")
    print(f"处理完一共有{len(nodes)}个节点")

    client = RetrieverClient()
    client.wait_service_available(embedding_model)
    #print(client.info(embedding_model))

    sentences = [mode["text"] for mode in nodes]
    embeddings = []

    l = len(sentences)
    for ndx in trange(0, l, batchsize):
        batch = sentences[ndx: min(ndx + batchsize, l)]
        e = client.encode(embedding_model, batch).vecs['dense_vecs']
        embeddings.append(e)
    embeddings = np.vstack(embeddings)

    pickle.dump(
        {"nodes": nodes, "embeddings": embeddings, "collection": collection, "embedding_model": embedding_model},
        open(f"{file.parent / (pickle_name + '.pkl')}", "wb")
    )
    return embeddings


if __name__ == '__main__':
    embeddings = text2vec(collection="test", embedding_model="BAAI/bge-m3")
    print(embeddings.shape)
