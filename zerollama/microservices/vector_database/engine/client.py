
from hashlib import md5
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.microservices.vector_database.protocol import PROTOCOL
from zerollama.microservices.vector_database.protocol import VectorDatabaseTopKRequest
from zerollama.microservices.vector_database.protocol import VectorDatabaseTopKResponse

CLIENT_VALIDATION = True


class VectorDatabaseClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    @staticmethod
    def get_db_name(collection, embedding_model):
        pickle_name = f"zerollama:{collection}:{embedding_model}:embeddings"
        name = md5(pickle_name.encode("utf-8")).hexdigest()
        return name

    def top_k(self, collection, embedding_model, query_dense_vecs, k=10):
        method = "top_k"
        data = {"embedding_model": embedding_model,
                "query_dense_vecs": query_dense_vecs,
                "k": k}
        if CLIENT_VALIDATION:
            data = VectorDatabaseTopKRequest(**data).dict()

        name = self.get_db_name(collection, embedding_model)

        rep = self.query(name, method, data)
        if rep is None:
            raise RuntimeError(f"VectorDatabase [{collection}] server not found.")

        if rep.state == "ok":
            rep = VectorDatabaseTopKResponse(**rep.msg)
        else:
            raise RuntimeError(f"VectorDatabase [{collection}] error, with error msg [{rep.msg}]")
        return rep


if __name__ == '__main__':
    import pickle
    CLIENT_VALIDATION = False

    from zerollama.core.config.main import config_setup

    config = config_setup()

    collection = "test"
    embedding_model = "BAAI/bge-m3"
    pickle_name = md5(f"zerollama:{collection}:{embedding_model}:embeddings".encode("utf-8")).hexdigest()
    pickle_file = f"{config.rag.path / collection / 'embeddings' / (pickle_name + '.pkl')}"

    client = VectorDatabaseClient()

    name = client.get_db_name(collection, embedding_model)

    print("=" * 80)
    print(f"Wait VectorDatabase available")

    client.wait_service_available(name)
    print(client.get_services(name))

    print("=" * 80)
    print('VectorDatabase support_methods')
    print(client.support_methods(name))
    print(client.info(name))

    data = pickle.load(open(pickle_file, "rb"))
    embeddings = data["embeddings"]

    top_k = client.top_k(collection, embedding_model, query_dense_vecs=embeddings[10], k=10)

    for n in top_k.data:
        print(n.score)
        print(n.index)
        print(n.node["text"])
