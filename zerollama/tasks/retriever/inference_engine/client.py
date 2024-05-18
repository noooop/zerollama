
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.retriever.protocol import PROTOCOL
from zerollama.tasks.retriever.protocol import RetrieverRequest, RetrieverResponse

CLIENT_VALIDATION = True


class RetrieverClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def encode(self, name, sentences, options=None):
        method = "inference"
        data = {"model": name,
                "sentences": sentences,
                "options": options or dict()}

        if CLIENT_VALIDATION:
            data = RetrieverRequest(**data).dict()

        rep = self.query(name, method, data)
        if rep is None:
            return rep

        if rep.state == "ok":
            rep = RetrieverResponse(**rep.msg)
        return rep


if __name__ == '__main__':
    CLIENT_VALIDATION = False

    model_name = "BAAI/bge-m3"

    client = RetrieverClient()
    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('ZeroRetrieverInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

    print("="*80)
    embeddings_1 = client.encode(model_name, sentences_1).vecs['dense_vecs']
    embeddings_2 = client.encode(model_name, sentences_2).vecs['dense_vecs']

    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)

