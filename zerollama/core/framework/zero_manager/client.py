

from zerollama.core.framework.nameserver.client import ZeroClient


class ZeroManagerClient(ZeroClient):
    protocol = "manager"

    def __init__(self, name):
        self.name = name
        ZeroClient.__init__(self, self.protocol)

    def start(self, model_name, model_class, model_kwargs):
        data = {
            "method": "start",
            "name": model_name,
            "model_class": model_class,
            "model_kwargs": model_kwargs,
        }
        return self.json_query(self.name, data)

    def terminate(self, model_name):
        data = {
            "method": "terminate",
            "name": model_name,
        }
        return self.json_query(self.name, data)

    def list(self):
        data = {
            "method": "list",
        }
        return self.json_query(self.name, data)


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

    model_names = ["Qwen/Qwen1.5-0.5B-Chat",
                   # "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",
                   "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",
                   "Qwen/Qwen1.5-0.5B-Chat-AWQ"]

    for model_name in model_names:
        model_class = "zerollama.models.qwen.qwen1_5:Qwen1_5"
        model_kwargs = {"model_name": model_name}

        print("=" * 80)
        print('start', model_name)
        print(manager_client.start(model_name, model_class, model_kwargs))
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
        client.wait_service_available(model_name)
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

    for model_name in model_names:
        print("=" * 80)
        print('terminate', model_name)
        print(manager_client.terminate(model_name))
        print(manager_client.list())
