import pickle
import numpy as np
from tqdm import trange
from hashlib import md5
from pathlib import Path
from zerollama.scaffold.documents.splitter.main import text_window_parser
from zerollama.tasks.retriever.engine.client import RetrieverClient
from zerollama.core.config.main import config_setup


def batchify(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def chunk(collection):
    config = config_setup()

    pickle_file = f"{config.rag.path / collection / 'chunk' / 'chunk.pkl'}"

    if Path(pickle_file).exists():
        data = pickle.load(open(pickle_file, "rb"))
        return data
    else:
        print(Path(pickle_file).parent)
        Path(pickle_file).parent.mkdir(exist_ok=True)

    nodes_list = []
    for file in (config.rag.path / collection / "text").glob("*.txt"):
        book_name = file.stem.split("-")[0]
        lines, sentence_list, nodes = text_window_parser(file)
        print(book_name)

        print(f"一共有{len(lines)}个段落")
        print(f"一共有{len(sentence_list)}个句子, 最大长度为{max(len(line['text']) for line in sentence_list)}")
        print(f"处理完一共有{len(nodes)}个节点")
        nodes_list.extend(nodes)

    data = {"collection": collection, "nodes": nodes_list}

    pickle.dump(data, open(pickle_file, "wb"))
    return data


def text2vec(collection, embedding_model, batchsize=128):
    config = config_setup()

    pickle_name = md5(f"zerollama:{collection}:{embedding_model}:embeddings".encode("utf-8")).hexdigest()
    pickle_file = f"{config.rag.path / collection / 'embeddings' / (pickle_name + '.pkl')}"

    data = chunk(collection)
    nodes = data["nodes"]
    sentences = [node["text"] for node in nodes]

    hash = md5("\n".join(sentences).encode("utf-8")).hexdigest()

    if Path(pickle_file).exists():
        data = pickle.load(open(pickle_file, "rb"))
        if data["hash"] == hash:
            return data["embeddings"]
    else:
        Path(pickle_file).parent.mkdir(exist_ok=True)

    client = RetrieverClient()
    client.wait_service_available(embedding_model)

    embeddings = []
    l = len(sentences)
    for ndx in trange(0, l, batchsize):
        batch = sentences[ndx: min(ndx + batchsize, l)]
        e = client.encode(embedding_model, batch).vecs['dense_vecs']
        embeddings.append(e)
    embeddings = np.vstack(embeddings)

    pickle.dump(
        {"nodes": nodes, "embeddings": embeddings, "collection": collection, "embedding_model": embedding_model, "hash": hash},
        open(pickle_file, "wb")
    )
    return embeddings


if __name__ == '__main__':
    embeddings = text2vec(collection="test_collection", embedding_model="BAAI/bge-m3")
    print(embeddings.shape)
