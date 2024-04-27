import time

from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.core.framework.zero_manager.protocol import StartRequest, TerminateRequest, StatusRequest

CLIENT_VALIDATION = True


class ZeroManagerClient(ZeroClient):
    protocol = "manager"

    def __init__(self, name):
        self.name = name
        ZeroClient.__init__(self, self.protocol)

    def start(self, model_name, model_class, model_kwargs):
        data = {"name": model_name,
                "model_class": model_class,
                "model_kwargs": model_kwargs}
        method = "start"

        if CLIENT_VALIDATION:
            data = StartRequest(**data).dict()

        return self.query(self.name, method, data)

    def terminate(self, model_name):
        method = "terminate"
        data = {
            "name": model_name,
        }
        if CLIENT_VALIDATION:
            data = TerminateRequest(**data).dict()
        return self.query(self.name, method, data)

    def list(self):
        method = "list"
        return self.query(self.name, method)

    def statuses(self):
        method = "statuses"
        return self.query(self.name, method)

    def status(self, model_name):
        method = "status"
        data = {
            "name": model_name,
        }
        if CLIENT_VALIDATION:
            data = StatusRequest(**data).dict()
        return self.query(self.name, method, data)


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

    model_class = "zerollama.inference_backend.hf_transformers.main:HuggingFaceTransformersChat"
    model_names = ["Qwen/Qwen1.5-0.5B-Chat-AWQ",
                   "openbmb/MiniCPM-2B-sft-bf16"]

    for model_name in model_names + ["not_found", "Qwen/Qwen1.5-72B-Chat"]:
        model_kwargs = {"model_name": model_name}

        print("=" * 80)
        print('start', model_name)
        print(manager_client.start(model_name, model_class, model_kwargs))
        print(manager_client.list())
        print(manager_client.statuses())

    """
    for i in range(100):
        time.sleep(0.5)
        rep = manager_client.statuses()
        for k in rep.msg.keys():
            print(manager_client.status(k))
    """

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

    for model_name in model_names + ["not_found", "Qwen/Qwen1.5-72B-Chat"]:
        print("=" * 80)
        print('terminate', model_name)
        print(manager_client.terminate(model_name))
        print(manager_client.list())