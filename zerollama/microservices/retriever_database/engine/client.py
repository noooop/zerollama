
from hashlib import md5
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.microservices.retriever_database.protocol import PROTOCOL
from zerollama.microservices.retriever_database.protocol import RetrieverDatabaseTopKRequest
from zerollama.microservices.retriever_database.protocol import RetrieverDatabaseTopKResponse

CLIENT_VALIDATION = True


class RetrieverDatabaseClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    @staticmethod
    def get_db_name(collection, retriever_model):
        pickle_name = f"zerollama:{collection}:{retriever_model}"
        name = md5(pickle_name.encode("utf-8")).hexdigest()
        return name

    def top_k(self, collection, retriever_model, query, k=10):
        method = "top_k"
        data = {"query": query,
                "retriever_model": retriever_model,
                "k": k}
        if CLIENT_VALIDATION:
            data = RetrieverDatabaseTopKRequest(**data).dict()

        name = self.get_db_name(collection, retriever_model)

        rep = self.query(name, method, data)
        if rep is None:
            raise RuntimeError(f"RetrieverDatabase [{collection}] server not found.")

        if not rep.state == "ok":
            raise RuntimeError(f"RetrieverDatabase [{collection}] error, with error msg [{rep.msg}]")

        return RetrieverDatabaseTopKResponse(**rep.msg)


if __name__ == '__main__':
    import pickle
    CLIENT_VALIDATION = False

    from zerollama.core.config.main import config_setup

    config = config_setup()

    collection = "test_collection"
    retriever_model = "BM25s"

    pickle_file = f"{config.rag.path / collection / 'chunk' / 'chunk.pkl'}"

    client = RetrieverDatabaseClient()
    name = client.get_db_name(collection, retriever_model)

    print("=" * 80)
    print(f"Wait RetrieverDatabase available")

    client.wait_service_available(name)
    print(client.get_services(name))

    print("=" * 80)
    print('RetrieverDatabase support_methods')
    print(client.support_methods(name))
    print(client.info(name))

    data = pickle.load(open(pickle_file, "rb"))
    query = data["nodes"][10]["text"]

    top_k = client.top_k(collection, retriever_model, query, k=10)

    for n in top_k.data:
        print(n.score)
        print(n.index)
        print(n.node["text"])
