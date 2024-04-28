from zerollama.core.framework.zero_manager.client import ZeroManagerClient

if __name__ == '__main__':
    from pprint import pprint
    name = "ZeroInferenceManager"
    manager_client = ZeroManagerClient(name)

    print("=" * 80)
    print(f"Wait {name} available")
    manager_client.wait_service_available(name)
    print(manager_client.get_service_names())

    print("=" * 80)
    print(f'{name} support_methods')
    print(manager_client.support_methods(name))

    print("=" * 80)
    print(f'init')
    print(manager_client.list())
    print(manager_client.statuses())

    protocol = "chat"
    model_kwargs = {}
    model_names = ["Qwen/Qwen1.5-0.5B-Chat-AWQ",
                   "openbmb/MiniCPM-2B-sft-bf16"]

    for model_name in model_names:
        print("=" * 80)
        print('start', model_name)
        print(manager_client.start(model_name, protocol, model_kwargs))
        print(manager_client.list())

    def test_inference_engine(model_name):
        prompt = "给我介绍一下大型语言模型。"
        messages = [
            {"role": "user", "content": prompt}
        ]

        from zerollama.core.framework.inference_engine.client import ChatClient

        client = ChatClient()
        print("=" * 80)
        print(f"Wait {model_name} available")
        manager_client.wait_service_status(model_name)
        print(client.get_services(model_name))

        print("=" * 80)
        print('ZeroInferenceEngine support_methods')
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

    #for model_name in model_names:
    #    print("=" * 80)
    #    print('terminate', model_name)
    #    print(manager_client.terminate(model_name))
    #    print(manager_client.list())