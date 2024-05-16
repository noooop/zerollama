
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.reranker.protocol import PROTOCOL
from zerollama.tasks.reranker.protocol import RerankerRequest, RerankerResponse

CLIENT_VALIDATION = True


class RerankerClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def compute_score(self, name, sentence_pairs, options=None):
        method = "inference"
        data = {"model": name,
                "sentence_pairs": sentence_pairs,
                "options": options or dict()}

        if CLIENT_VALIDATION:
            data = RerankerRequest(**data).dict()

        rep = self.query(name, method, data)
        if rep is None:
            return rep

        if rep.state == "ok":
            rep = rep.unload()
            rep = RerankerResponse(**rep)
        return rep


if __name__ == '__main__':
    CLIENT_VALIDATION = False

    model_name = "BAAI/bge-reranker-v2-m3"

    client = RerankerClient()
    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('ZeroRetrieverInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    sentence_pairs = [['what is panda?', 'hi'],
                      ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

    print("="*80)
    scores = client.compute_score(model_name, sentence_pairs).vecs["scores"].tolist()

    print(scores)

