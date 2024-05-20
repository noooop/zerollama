from zerollama.core.framework.zero_manager.client import ZeroManagerClient
from zerollama.tasks.chat.protocol import MANAGER_NAME


def test(x):
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

    model_names = ["Qwen/Qwen1.5-0.5B-Chat-AWQ"]

    for model_name in model_names:
        print("=" * 80)
        print('start', model_name)
        print(manager_client.start(model_name))
        print(manager_client.list())

    def test_inference_engine(model_name):
        prompt = "给我介绍一下大型语言模型。"
        messages = [
            {"role": "user", "content": prompt}
        ]

        from zerollama.tasks.chat.engine.client import ChatClient

        client = ChatClient()
        print("=" * 80)
        print(f"Wait {model_name} available")
        manager_client.wait_service_status(model_name)
        print(client.get_services(model_name))

        print("=" * 80)
        print('ZeroChatInferenceEngine support_methods')
        print(client.support_methods(model_name))

        print("="*80)
        print("stream == False")
        msg = client.chat(model_name, messages)
        pprint(msg)

        print("="*80)
        print("stream == True")
        for msg in client.stream_chat(model_name, messages):
            pprint(msg)

    for model_name in model_names:
        test_inference_engine(model_name)


if __name__ == '__main__':
    import sys
    from multiprocessing import Pool

    if len(sys.argv) == 1:
        n = 1
    else:
        n = int(sys.argv[1])
    p = Pool(n)
    for x in p.imap_unordered(test, range(n)):
        pass


