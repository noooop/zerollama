
import inspect
import requests
from functools import partial
from zerollama.core.config.main import config_setup
from zerollama.tasks.chat.interface import ChatInterface
from zerollama.tasks.chat.protocol import ChatCompletionResponse
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponse, ChatCompletionStreamResponseDone
from zerollama.tasks.chat.collection import get_model_config_by_name


class LLamaCPPChat(ChatInterface):
    def __init__(self, model_name, local_files_only=True, verbose=False, n_ctx=1000, device="cuda"):
        model_config = get_model_config_by_name(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.verbose = verbose
        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.local_files_only = local_files_only
        self.n_ctx = n_ctx
        self.model = None
        self.n_concurrent = 1

    def load(self):
        config_setup()

        if self.local_files_only:
            import huggingface_hub
            huggingface_hub.snapshot_download = partial(huggingface_hub.snapshot_download,
                                                        local_files_only=True)

        from llama_cpp import Llama
        repo_id, filename = self.model_name.split("+")

        try:
            model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                verbose=self.verbose,

                n_gpu_layers=-1,
                n_ctx=self.n_ctx
            )
        except requests.exceptions.HTTPError:
            raise ValueError(f"model '{self.model_name}' not found, try pulling it first") from None
        except EnvironmentError:
            raise ValueError(f"model '{self.model_name}' not found, try pulling it first") from None

        self.model = model

    @property
    def info(self):
        return self.model_info

    def chat(self, messages, stream=False, options=None):
        options = options or dict()
        response = self.model.create_chat_completion(messages, stream=stream, **options)

        if not inspect.isgenerator(response):
            return ChatCompletionResponse(**{"model": self.model_name,
                                             "content": response['choices'][0]['message']['content'],
                                             "finish_reason": response['choices'][0]['finish_reason'],

                                             "completion_tokens": response['usage']['completion_tokens'],
                                             "prompt_tokens": response['usage']['prompt_tokens'],
                                             "total_tokens": response['usage']['total_tokens']})
        else:
            def generator():
                completion_tokens = 0
                for rep in response:
                    completion_tokens += 1

                    finish_reason = rep['choices'][0]['finish_reason']

                    if finish_reason is None:
                        content = rep['choices'][0]['delta'].get("content", None)
                        if content is None:
                            continue

                        yield ChatCompletionStreamResponse(**{"model": self.model_name,
                                                              "delta_content": content})
                    else:
                        yield ChatCompletionStreamResponseDone(**{"model": self.model_name,
                                                                  "finish_reason": finish_reason,

                                                                  "prompt_tokens": -1,
                                                                  "completion_tokens": completion_tokens,
                                                                  "total_tokens": -1})
                        break

            return generator()


def run_test(model_name, stream=False):
    from pprint import pprint
    print("=" * 80)

    model = LLamaCPPChat(model_name, local_files_only=False)
    model.load()
    pprint(model.info)
    print("=" * 80)

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = model.chat(messages, options={"max_tokens": 512}, stream=stream)

    if inspect.isgenerator(response):
        for part in response:
            if isinstance(part, ChatCompletionStreamResponseDone):
                print()
                print("completion_tokens:", part.completion_tokens)
            else:
                print(part.delta_content, end="", flush=True)
    else:
        print(response.content)
        print("completion_tokens:", response.completion_tokens)


if __name__ == '__main__':
    model_name = "Qwen/Qwen1.5-0.5B-Chat-GGUF+*q8_0.gguf"

    run_test(model_name, stream=False)
    run_test(model_name, stream=True)