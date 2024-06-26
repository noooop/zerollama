
from zerollama.microservices.vector_database.interface import VectorDatabaseInterface
from zerollama.microservices.vector_database.protocol import TopKNode, VectorDatabaseTopKResponse


class FaissVectorDatabase(VectorDatabaseInterface):
    Index = "IndexHNSWFlat"

    M = 64  # number of connections each vertex will have
    ef_search = 32  # depth of layers explored during search
    ef_construction = 64  # depth of layers explored during index construction

    def init(self):
        import faiss

        if self.Index == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        elif self.Index == "IndexHNSWFlat":
            self.index = faiss.IndexHNSWFlat(self.embeddings.shape[1], self.M)
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search

        self.index.add(self.embeddings)

    def top_k(self, query_dense_vecs, embedding_model, k=10):
        if embedding_model != self.embedding_model:
            raise ValueError(f"[{embedding_model}] not support")

        if self.Index == "IndexHNSWFlat":
            distances, index = self.index.search(query_dense_vecs[None, :], k)
            data = [TopKNode(index=i, score=1-score, node=self.nodes[int(i)])
                    for i, score in zip(index[0], distances[0])]

        elif self.Index == "IndexFlatIP":
            distances, index = self.index.search(query_dense_vecs[None, :], k)

            data = [TopKNode(index=i, score=score, node=self.nodes[int(i)])
                    for i, score in zip(index[0], distances[0])]
        else:
            raise ValueError(f"{self.Index} not support")

        return VectorDatabaseTopKResponse(embedding_model=self.embedding_model, data=data)


if __name__ == '__main__':
    from hashlib import md5
    from zerollama.core.config.main import config_setup

    config = config_setup()

    collection = "test"
    embedding_model = "BAAI/bge-m3"
    pickle_name = md5(f"zerollama:{collection}:{embedding_model}:embeddings".encode("utf-8")).hexdigest()
    pickle_file = f"{config.rag.path / collection / 'embeddings' / (pickle_name + '.pkl')}"

    vdb = FaissVectorDatabase.load_from_file(pickle_file)

    top_k = vdb.top_k(embedding_model=embedding_model, query_dense_vecs=vdb.embeddings[10], k=10)

    for n in top_k.data:
        print(n.score)
        print(n.index)
        print(n.node["text"])