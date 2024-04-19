
from zerollama.core.framework.nameserver.client import ZeroClient


class ChatClient(ZeroClient):
    protocol = "chat"

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def chat(self, name, messages, options=None, **kwargs):
        data = {
            "method": "inference",
            "model": name,
            "messages": messages,
            "options": options,
            "stream": False
        }
        return self.json_query(name, data, **kwargs)

    def stream_chat(self, name, messages, options=None, **kwargs):
        data = {
            "method": "inference",
            "model": name,
            "messages": messages,
            "options": options,
            "stream": True
        }

        for part in self.json_stream_query(name, data, **kwargs):
            yield part


if __name__ == '__main__':
    from pprint import pprint

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    model_name = "Qwen/Qwen1.5-0.5B-Chat"

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