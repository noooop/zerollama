from zerollama.core.framework.zero_manager.client import ZeroManagerClient
from zerollama.tasks.retriever.protocol import MANAGER_NAME

if __name__ == '__main__':
    from pprint import pprint
    manager_client = ZeroManagerClient(MANAGER_NAME)

    print("=" * 80)
    print(f"Wait {MANAGER_NAME} available")
    manager_client.wait_service_available(MANAGER_NAME)
    print(manager_client.get_service_names())

    print("=" * 80)
    print(f'{MANAGER_NAME} support_methods')
    print(manager_client.support_methods(MANAGER_NAME))

    print("=" * 80)
    print(f'init')
    print(manager_client.list())
    print(manager_client.statuses())

    model_names = ["BAAI/bge-m3"]

    for model_name in model_names:
        print("=" * 80)
        print('start', model_name)
        print(manager_client.start(model_name))
        print(manager_client.list())

    def test_inference_engine(model_name):
        sentences_1 = ["What is BGE M3?", "Defination of BM25"]
        sentences_2 = [
            "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
            "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

        from zerollama.tasks.retriever.inference_engine.client import RetrieverClient

        client = RetrieverClient()
        print("=" * 80)
        print(f"Wait {model_name} available")
        manager_client.wait_service_status(model_name)
        print(client.get_services(model_name))

        print("=" * 80)
        print('ZeroRetrieverInferenceEngine support_methods')
        print(client.support_methods(model_name))

        embeddings_1 = client.encode(model_name, sentences_1).vecs['dense_vecs']
        embeddings_2 = client.encode(model_name, sentences_2).vecs['dense_vecs']

        print("=" * 80)
        similarity = embeddings_1 @ embeddings_2.T
        print(similarity)


    for model_name in model_names:
        print(model_name)
        test_inference_engine(model_name)