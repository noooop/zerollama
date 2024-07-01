import gc
import traceback

import torch
import requests
import shortuuid

from zerollama.tasks.chat.interface import ChatInterface
from zerollama.tasks.chat.protocol import ChatCompletionResponse
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponse, ChatCompletionStreamResponseDone
from zerollama.tasks.chat.collection import get_model_by_name
from zerollama.tasks.base.download import get_pretrained_model_name_or_path


class MIIChat(ChatInterface):
    get_model_by_name = staticmethod(get_model_by_name)

    def __init__(self, model_name, local_files_only=True, device="cuda", **engine_args):
        model = self.get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.local_files_only = local_files_only
        self.trust_remote_code = self.model_config.model_kwargs.get("trust_remote_code", False)
        self.pretrained_model_name_or_path = get_pretrained_model_name_or_path(model_name=model_name,
                                                                               local_files_only=local_files_only,
                                                                               get_model_by_name=get_model_by_name)
        self.engine_args = engine_args
        self.engine = None
        self.TextTokensPrompt = None
        self.n_concurrent = 1

    def load(self):
        from mii import pipeline

        try:
            engine = pipeline(model_name_or_path=str(self.pretrained_model_name_or_path))
        except requests.exceptions.HTTPError:
            traceback.print_exc()
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first") from None
        except EnvironmentError:
            traceback.print_exc()
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first") from None

        self.engine = engine

    def __del__(self):
        self.engine = None

        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def info(self):
        return self.model_info

    def chat(self, messages, stream=False, options=None):
        options = options or {}
        options["max_length"] = options.pop("max_tokens", 256)

        tokenizer = self.engine.tokenizer.tokenizer
        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        responses = self.engine(prompts=[prompt], **options)
        response = responses[0]

        return ChatCompletionResponse(**{"model": self.model_name,
                                         "content": response.generated_text,
                                         "finish_reason": response.finish_reason,

                                         "completion_tokens": response.generated_length,
                                         "prompt_tokens": response.prompt_length,
                                         "total_tokens": response.prompt_length + response.generated_length})


def run_test(model_name, stream=False, **kwargs):
    import inspect
    print("=" * 80)

    model = MIIChat(model_name, local_files_only=False, **kwargs)
    model.load()
    print(model.info)

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = model.chat(messages, stream=stream)

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
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    run_test(model_name, stream=False)