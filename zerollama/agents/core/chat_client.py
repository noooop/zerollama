
import inspect
from zerollama.tasks.chat.interface import ChatInterface
from zerollama.tasks.chat.protocol import ChatCompletionResponse
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponse, ChatCompletionStreamResponseDone


class ZerollamaChatClient(ChatInterface):
    def __init__(self, model, options=None):
        from zerollama.tasks.chat.engine.client import ChatClient
        self.options = options or {}
        self._chat_client = ChatClient()
        self.model_name = model

    def chat(self, messages, tools=None, stream=False, options=None):
        options = dict(**self.options)
        options.update(options or {})
        response = self._chat_client.chat(self.model_name, messages, tools, stream, options)
        return response


class OpenAiChatClient(ChatInterface):
    def __init__(self, model, options=None, base_url="http://localhost:8080", api_key="empty", **kwargs):
        from gevent import monkey
        monkey.patch_socket()

        from openai import OpenAI, Stream
        self.options = options or {}

        self.Stream = Stream
        self._chat_client = OpenAI(base_url=base_url, api_key=api_key, **kwargs)
        self.model_name = model

    def chat(self, messages, tools=None, stream=False, options=None):
        options = dict(**self.options)
        options.update(options or {})

        response = self._chat_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            tools=tools,
            **options
        )

        if not isinstance(response, self.Stream):
            return ChatCompletionResponse(**{"model": self.model_name,
                                             "content": response.choices[0].message.content,
                                             "finish_reason": response.choices[0].finish_reason,

                                             "completion_tokens": response.usage.completion_tokens,
                                             "prompt_tokens": response.usage.prompt_tokens,
                                             "total_tokens": response.usage.total_tokens})
        else:
            def generator():

                for res in response:
                    res = res.choices[0]
                    if res.finish_reason is None:
                        yield ChatCompletionStreamResponse(**{"model": self.model_name,
                                                              "delta_content": res.delta.content})
                    else:
                        yield ChatCompletionStreamResponseDone(**{"model": self.model_name,
                                                                  "finish_reason": res.finish_reason,

                                                                  "completion_tokens": 0,
                                                                  "prompt_tokens": 0,
                                                                  "total_tokens": 0})

            return generator()


class OllamaChatClient(ChatInterface):
    def __init__(self, model, options=None, base_url=None, **kwargs):
        from gevent import monkey
        monkey.patch_socket()

        from ollama import Client
        self.options = options or {}
        self._chat_client = Client(host=base_url, **kwargs)
        self.model_name = model

    def chat(self, messages, tools=None, stream=False, options=None):
        options = dict(**self.options)
        options.update(options or {})

        response = self._chat_client.chat(
            model=self.model_name,
            messages=messages,
            stream=stream,
            options=options
        )

        if not inspect.isgenerator(response):
            return ChatCompletionResponse(**{"model": self.model_name,
                                             "content": response['message']['content'],
                                             "finish_reason": response['done_reason'],

                                             "completion_tokens": response.get('eval_count', 0),
                                             "prompt_tokens": response.get('prompt_eval_count', 0),
                                             "total_tokens": response.get('eval_count', 0)+response.get('prompt_eval_count', 0)})
        else:
            def generator():
                for res in response:
                    if not res["done"]:
                        yield ChatCompletionStreamResponse(**{"model": self.model_name,
                                                              "delta_content": res['message']['content']})
                    else:
                        yield ChatCompletionStreamResponseDone(**{"model": self.model_name,
                                                                  "finish_reason": res['done_reason'],

                                                                  "completion_tokens": res.get('eval_count', 0),
                                                                  "prompt_tokens": res.get('prompt_eval_count', 0),
                                                                  "total_tokens": res.get('eval_count', 0) + res.get('prompt_eval_count', 0)})

            return generator()


def get_client(llm_config=None, global_priority=None):
    import copy
    from zerollama.core.config.main import config_setup
    config = config_setup()
    global_llm_config = config.get("llm_config", {})

    if isinstance(llm_config, dict) and "global_priority" in llm_config:
        global_priority = llm_config.pop("global_priority")

    if global_llm_config:
        if global_priority is None:
            global_priority = global_llm_config.pop("global_priorit", True)

        if global_priority:
            llm_config = global_llm_config
        elif llm_config is None or not llm_config:
            llm_config = global_llm_config

    llm_config = copy.deepcopy(llm_config)
    llm_client_type = llm_config.pop("type")

    if llm_client_type == "zerollama":
        client = ZerollamaChatClient(**llm_config)
    elif llm_client_type == "openai":
        client = OpenAiChatClient(**llm_config)
    elif llm_client_type == "ollama":
        client = OllamaChatClient(**llm_config)
    else:
        raise KeyError(f"llm client type {llm_client_type} not support")

    return client


if __name__ == '__main__':
    model = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"
    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    for llm_config in [
        {"type": "zerollama", "model": model},
        {"type": "openai", "model": model, "base_url": 'http://localhost:8080/v1/'},
        {"type": "ollama", "model": model}

    ]:
        print("=" * 80)
        print(llm_config)

        client = get_client(llm_config, global_priority=False)

        print("stream=False")
        print(client.chat(messages).content)

        print("stream=True")
        for part in client.chat(messages, stream=True):
            if isinstance(part, ChatCompletionStreamResponseDone):
                print()
                print("completion_tokens:", part.completion_tokens)
            else:
                print(part.delta_content, end="", flush=True)