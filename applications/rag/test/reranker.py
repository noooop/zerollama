from zerollama.core.framework.zero_manager.client import ZeroManagerClient
from zerollama.tasks.reranker.protocol import MANAGER_NAME

if __name__ == '__main__':
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

    model_names = ["BAAI/bge-reranker-v2-m3"]

    for model_name in model_names:
        print("=" * 80)
        print('start', model_name)
        print(manager_client.start(model_name))
        print(manager_client.list())

    def test_inference_engine(model_name):
        sentence_pairs = [['what is panda?', 'hi'],
                          ['what is panda?',
                           'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

        from zerollama.tasks.reranker.engine.client import RerankerClient

        client = RerankerClient()
        print("=" * 80)
        print(f"Wait {model_name} available")
        manager_client.wait_service_status(model_name)
        print(client.get_services(model_name))

        print("=" * 80)
        print('ZeroRerankerInferenceEngine support_methods')
        print(client.support_methods(model_name))

        print("=" * 80)
        scores = client.compute_score(model_name, sentence_pairs).vecs["scores"].tolist()

        print(scores)


    for model_name in model_names:
        print(model_name)
        test_inference_engine(model_name)