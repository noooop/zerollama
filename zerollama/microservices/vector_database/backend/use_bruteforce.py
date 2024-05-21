

import numpy as np
from zerollama.microservices.vector_database.interface import VectorDatabaseInterface
from zerollama.microservices.vector_database.protocol import TopKNode, VectorDatabaseTopKResponse


class BruteForceVectorDatabase(VectorDatabaseInterface):

    def init(self):
        pass

    def top_k(self, query_dense_vecs, embedding_model, k=10):
        if embedding_model != self.embedding_model:
            raise ValueError(f"[{embedding_model}] not support")

        scores = query_dense_vecs @ self.embeddings.T
        index = np.argsort(scores)

        data = [TopKNode(index=i, score=scores[i], node=self.nodes[i]) for i in reversed(index[-k:])]
        return VectorDatabaseTopKResponse(embedding_model=self.embedding_model, data=data)


if __name__ == '__main__':
    from hashlib import md5
    from zerollama.core.config.main import config_setup

    config = config_setup()

    collection = "test"
    embedding_model = "BAAI/bge-m3"
    pickle_name = md5(f"zerollama:{collection}:{embedding_model}:embeddings".encode("utf-8")).hexdigest()
    pickle_file = f"{config.rag.path / collection / 'embeddings' / (pickle_name + '.pkl')}"

    vdb = BruteForceVectorDatabase.load_from_file(pickle_file)

    top_k = vdb.top_k(embedding_model=embedding_model, query_dense_vecs=vdb.embeddings[10], k=10)

    for n in top_k.data:
        print(n.score)
        print(n.index)
        print(n.node["text"])