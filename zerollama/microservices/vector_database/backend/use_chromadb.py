
from zerollama.microservices.vector_database.interface import VectorDatabaseInterface
from zerollama.microservices.vector_database.protocol import TopKNode, VectorDatabaseTopKResponse


class ChromadbVectorDatabase(VectorDatabaseInterface):
    def init(self):
        import chromadb
        self.index = chromadb.Client()
        self.collection = self.index.create_collection(name="collection",
                                                       metadata={"hnsw:space": "ip"} )

        self.collection.add(
            embeddings=self.embeddings,
            documents=[node["text"] for node in self.nodes],
            ids=[f"id-{i:06d}" for i in range(len(self.nodes))]
        )

        #self.embeddings = None
        self.nodes = None

    def top_k(self, query_dense_vecs, embedding_model, k=10):
        if embedding_model != self.embedding_model:
            raise ValueError(f"[{embedding_model}] not support")

        results = self.collection.query(
            query_embeddings=query_dense_vecs.tolist(),
            n_results=k,
            include=['documents', 'distances']
        )

        data = []
        for index, score, text in zip(results["ids"][0], results["distances"][0], results["documents"][0]):
            index = int(index.split("-")[-1])
            data.append(TopKNode(index=index, score=1-score, node={"text": text}))

        return VectorDatabaseTopKResponse(embedding_model=self.embedding_model, data=data)


if __name__ == '__main__':
    from hashlib import md5
    from zerollama.core.config.main import config_setup

    config = config_setup()

    collection = "test"
    embedding_model = "BAAI/bge-m3"
    pickle_name = md5(f"zerollama:{collection}:{embedding_model}:embeddings".encode("utf-8")).hexdigest()
    pickle_file = f"{config.rag.path / collection / 'embeddings' / (pickle_name + '.pkl')}"

    vdb = ChromadbVectorDatabase.load_from_file(pickle_file)

    top_k = vdb.top_k(embedding_model=embedding_model, query_dense_vecs=vdb.embeddings[10], k=10)

    for n in top_k.data:
        print(n.score)
        print(n.index)
        print(n.node["text"])