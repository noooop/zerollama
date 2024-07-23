
import inspect
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.chat.protocol import PROTOCOL
from zerollama.tasks.chat.protocol import ChatCompletionRequest, ChatCompletionResponse
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponse, ChatCompletionStreamResponseDone

CLIENT_VALIDATION = True


class ChatClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def chat(self, name, messages, tools=None, stream=False, options=None):
        method = "inference"
        data = {"model": name,
                "messages": messages,
                "tools": tools,
                "options": options or dict(),
                "stream": stream}
        if CLIENT_VALIDATION:
            data = ChatCompletionRequest(**data).dict()

        response = self.query(name, method, data)
        if response is None:
            raise RuntimeError(f"Chat [{name}] server not found.")

        if not inspect.isgenerator(response):
            if response.state != "ok":
                raise RuntimeError(f"Chat [{name}] error, with error msg [{response.msg}]")

            rep = ChatCompletionResponse(**response.msg)
            return rep
        else:
            def generator():
                for rep in response:
                    if rep is None:
                        raise RuntimeError(f"Chat [{name}] server not found.")

                    if rep.state != "ok":
                        raise RuntimeError(f"Chat [{name}] error, with error msg [{rep.msg}]")

                    if rep.msg["finish_reason"] is None:
                        rep = ChatCompletionStreamResponse(**rep.msg)
                    else:
                        rep = ChatCompletionStreamResponseDone(**rep.msg)

                    yield rep
            return generator()

    def tool_use(self, name, messages, tools, stream=False, options=None):
        method = "inference"
        data = {"model": name,
                "messages": messages,
                "options": options or dict(),
                "stream": stream}
        if CLIENT_VALIDATION:
            data = ChatCompletionRequest(**data).dict()

        response = self.query(name, method, data)
        if response is None:
            raise RuntimeError(f"Chat [{name}] server not found.")

        if not inspect.isgenerator(response):
            if response.state != "ok":
                raise RuntimeError(f"Chat [{name}] error, with error msg [{response.msg}]")

            rep = ChatCompletionResponse(**response.msg)
            return rep
        else:
            def generator():
                for rep in response:
                    if rep is None:
                        raise RuntimeError(f"Chat [{name}] server not found.")

                    if rep.state != "ok":
                        raise RuntimeError(f"Chat [{name}] error, with error msg [{rep.msg}]")

                    if rep.msg["finish_reason"] is None:
                        rep = ChatCompletionStreamResponse(**rep.msg)
                    else:
                        rep = ChatCompletionStreamResponseDone(**rep.msg)

                    yield rep
            return generator()

    def stream_chat(self, name, messages, options=None):
        for rep in self.chat(name, messages, stream=True, options=options):
            yield rep


if __name__ == '__main__':
    import shortuuid
    from pprint import pprint
    from gevent.pool import Pool
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
    print('ZeroChatInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    print("="*80)
    print("stream == False")
    response = client.chat(model_name, messages)
    print(response.content)
    print("response_length:", response.completion_tokens)

    print("="*80)
    print("stream == True")
    for msg in client.stream_chat(model_name, messages):
        pprint(msg)

    print("=" * 10)
    print("Test asynchronous")

    def worker(prompt):
        request_id = f"{shortuuid.random(length=22)}"
        messages = [
            {"role": "user", "content": prompt}
        ]
        generated_text = ""
        for output in client.stream_chat(model_name, messages):
            if not isinstance(output, ChatCompletionStreamResponseDone):
                generated_text += output.delta_content
                print(f"ID:{request_id}, Generated text: {generated_text!r}")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    p = Pool(32)
    for x in p.imap_unordered(worker, prompts*100):
        pass