
from hashlib import md5
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.workflow.vector_database.protocol import PROTOCOL
from zerollama.workflow.vector_database.protocol import VectorDatabaseTopKRequest
from zerollama.workflow.vector_database.protocol import VectorDatabaseTopKResponse

CLIENT_VALIDATION = True


class VectorDatabaseClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def get_db_name(self, filename, embedding_model):
        pickle_name = f"zerollama:{filename}:{embedding_model}:embeddings"
        name = md5(pickle_name.encode("utf-8")).hexdigest()
        return name

    def top_k(self, filename, embedding_model, query_dense_vecs, k=10):
        method = "top_k"
        data = {"embedding_model": embedding_model,
                "query_dense_vecs": query_dense_vecs,
                "k": k}
        if CLIENT_VALIDATION:
            data = VectorDatabaseTopKRequest(**data).dict()

        name = self.get_db_name(filename, embedding_model)

        rep = self.query(name, method, data)
        if rep is None:
            return rep

        if rep.state == "ok":
            rep = VectorDatabaseTopKResponse(**rep.msg)
        return rep


if __name__ == '__main__':
    import pickle
    from pathlib import Path
    CLIENT_VALIDATION = False

    rag_path = Path.home() / ".zerollama/rag/documents"
    filename = "test"
    embedding_model = "BAAI/bge-m3"
    file = list((rag_path / filename).glob("*.txt"))[0]
    book_name = file.stem.split("-")[0]

    client = VectorDatabaseClient()

    name = client.get_db_name(filename, embedding_model)

    print("=" * 80)
    print(f"Wait VectorDatabase available")

    client.wait_service_available(name)
    print(client.get_services(name))

    print("=" * 80)
    print('VectorDatabase support_methods')
    print(client.support_methods(name))
    print(client.info(name))

    data = pickle.load(open(f"{file.parent / (name + '.pkl')}", "rb"))
    embeddings = data["embeddings"]

    top_k = client.top_k(filename, embedding_model, query_dense_vecs=embeddings[10], k=10)

    for n in top_k.data:
        print(n.score)
        print(n.index)
        print(n.node["text"])
