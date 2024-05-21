
from zerollama.microservices.vector_database.interface import VectorDatabaseInterface
from zerollama.microservices.vector_database.protocol import TopKNode, VectorDatabaseTopKResponse


class HnswlibVectorDatabase(VectorDatabaseInterface):
    ef = 500
    M = 32
    num_threads = 4

    def init(self):
        import hnswlib

        self.index = hnswlib.Index(space='cosine', dim=self.embeddings.shape[1])
        self.index.init_index(max_elements=self.embeddings.shape[0], ef_construction=self.ef, M=self.M)
        self.index.set_num_threads(self.num_threads)

        self.index.add_items(self.embeddings)

    def top_k(self, query_dense_vecs, embedding_model, k=10):
        if embedding_model != self.embedding_model:
            raise ValueError(f"[{embedding_model}] not support")

        labels, scores = self.index.knn_query(query_dense_vecs, k=k)

        data = [TopKNode(index=index, score=1-score, node=self.nodes[index])
                for index, score in zip(labels[0], scores[0])]
        return VectorDatabaseTopKResponse(embedding_model=self.embedding_model, data=data)


if __name__ == '__main__':
    from hashlib import md5
    from zerollama.core.config.main import config_setup

    config = config_setup()

    collection = "test"
    embedding_model = "BAAI/bge-m3"
    pickle_name = md5(f"zerollama:{collection}:{embedding_model}:embeddings".encode("utf-8")).hexdigest()
    pickle_file = f"{config.rag.path / collection / 'embeddings' / (pickle_name + '.pkl')}"

    vdb = HnswlibVectorDatabase.load_from_file(pickle_file)

    top_k = vdb.top_k(embedding_model=embedding_model, query_dense_vecs=vdb.embeddings[10], k=10)

    for n in top_k.data:
        print(n.score)
        print(n.index)
        print(n.node["text"])