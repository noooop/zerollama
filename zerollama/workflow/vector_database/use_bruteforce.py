

import numpy as np
from zerollama.workflow.vector_database.interface import VectorDatabaseInterface
from zerollama.workflow.vector_database.protocol import VectorDatabaseTopKRequest
from zerollama.workflow.vector_database.protocol import TopKNode, VectorDatabaseTopKResponse


class BruteForceVectorDatabase(VectorDatabaseInterface):

    def init(self):
        pass

    def top_k(self, req: VectorDatabaseTopKRequest):
        if req.embedding_model != self.embedding_model:
            raise ValueError(f"[{req.embedding_model}] not support")

        scores = req.query_dense_vecs @ self.embeddings.T
        index = np.argsort(scores)

        data = [TopKNode(index=i, score=scores[i], node=self.nodes[i]) for i in reversed(index[-req.k:])]
        return VectorDatabaseTopKResponse(embedding_model=self.embedding_model, data=data)


if __name__ == '__main__':
    from hashlib import md5
    from pathlib import Path

    rag_path = Path.home() / ".zerollama/rag/documents"
    filename = "test"
    embedding_model = "BAAI/bge-m3"
    file = list((rag_path / filename).glob("*.txt"))[0]
    book_name = file.stem.split("-")[0]

    pickle_name = md5(f"zerollama:{file.stem}:{embedding_model}:embeddings".encode("utf-8")).hexdigest()

    vdb = BruteForceVectorDatabase.load_from_file(f"{file.parent / (pickle_name + '.pkl')}")

    req = VectorDatabaseTopKRequest(embedding_model=embedding_model, query_dense_vecs=vdb.embeddings[10], k=10)

    top_k = vdb.top_k(req)

    for n in top_k.data:
        print(n.score)
        print(n.index)
        print(n.node["text"])