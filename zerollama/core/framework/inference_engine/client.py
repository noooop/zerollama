
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.core.framework.inference_engine.protocol import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse

CLIENT_VALIDATION = True


class ChatClient(ZeroClient):
    protocol = "chat"

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def chat(self, name, messages, options=None):
        method = "inference"
        data = {"model": name,
                "messages": messages,
                "options": options or dict(),
                "stream": False}
        if CLIENT_VALIDATION:
            data = ChatCompletionRequest(**data).dict()

        rep = self.query(name, method, data)
        if rep.state == "ok":
            rep.msg = ChatCompletionResponse(**rep.msg)
        return rep

    def stream_chat(self, name, messages, options=None, **kwargs):
        method = "inference"
        data = {"model": name,
                "messages": messages,
                "options": options or dict(),
                "stream": True}
        if CLIENT_VALIDATION:
            data = ChatCompletionRequest(**data).dict()

        for rep in self.stream_query(name, method, data, **kwargs):
            if rep.state == "ok":
                rep.msg = ChatCompletionStreamResponse(**rep.msg)
            yield rep


if __name__ == '__main__':
    from pprint import pprint
    CLIENT_VALIDATION = False

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
    print(client.info(model_name))

    print("="*80)
    print("stream == False")
    response = client.chat(model_name, messages)
    print(response.msg.content)
    print("response_length:", response.msg.response_length)

    print("="*80)
    print("stream == True")
    for msg in client.stream_chat(model_name, messages):
        pprint(msg)