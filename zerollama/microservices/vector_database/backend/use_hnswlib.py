
from zerollama.microservices.vector_database.interface import VectorDatabaseInterface
from zerollama.microservices.vector_database.protocol import VectorDatabaseTopKRequest
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

    def top_k(self, req: VectorDatabaseTopKRequest):
        if req.embedding_model != self.embedding_model:
            raise ValueError(f"[{req.embedding_model}] not support")

        labels, scores = self.index.knn_query(req.query_dense_vecs, k=req.k)

        data = [TopKNode(index=index, score=1-score, node=self.nodes[index])
                for index, score in zip(labels[0], scores[0])]
        return VectorDatabaseTopKResponse(embedding_model=self.embedding_model, data=data)


if __name__ == '__main__':
    from hashlib import md5
    from zerollama.core.config.main import config_setup

    config = config_setup()

    collection = "test"
    embedding_model = "BAAI/bge-m3"
    file = list((config.rag.path / collection).glob("*.txt"))[0]
    book_name = file.stem.split("-")[0]

    pickle_name = md5(f"zerollama:{collection}:{embedding_model}:embeddings".encode("utf-8")).hexdigest()

    vdb = HnswlibVectorDatabase.load_from_file(f"{file.parent / (pickle_name + '.pkl')}")

    req = VectorDatabaseTopKRequest(embedding_model=embedding_model, query_dense_vecs=vdb.embeddings[10], k=10)

    top_k = vdb.top_k(req)

    for n in top_k.data:
        print(n.score)
        print(n.index)
        print(n.node["text"])