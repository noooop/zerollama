
import gc
import traceback

import torch
import requests
from threading import Thread
from functools import partial
from zerollama.core.config.main import config_setup
from zerollama.tasks.chat.interface import ChatInterface
from zerollama.tasks.chat.protocol import ChatCompletionResponse
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponse, ChatCompletionStreamResponseDone
from zerollama.tasks.chat.collection import get_model_by_name
from zerollama.tasks.base.download import get_pretrained_model_name

TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16


class HuggingFaceTransformers(object):
    get_model_by_name = staticmethod(get_model_by_name)

    def __init__(self, model_name, local_files_only=True, quantization_config=None, device="cuda"):
        model = self.get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.local_files_only = local_files_only
        self.quantization_config = quantization_config
        self.trust_remote_code = self.model_config.model_kwargs.get("trust_remote_code", False)
        self.pretrained_model_name = get_pretrained_model_name(model_name=model_name,
                                                               local_files_only=local_files_only,
                                                               get_model_by_name=get_model_by_name)
        self.torch_dtype = None

        self.model = None
        self.tokenizer = None
        self.streamer = None
        self.n_concurrent = 1

    def load(self):
        config = config_setup()

        if config.use_modelscope and not self.model_info.get("use_hf_only", False):
            from modelscope import AutoModelForCausalLM, AutoTokenizer
            from transformers import TextIteratorStreamer, BitsAndBytesConfig
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

        torch_dtype = "auto"
        if "quantization" in self.info and self.info["quantization"] != "":
            torch_dtype = torch.float16

        if self.quantization_config is not None:
            torch_dtype = torch.float16

        if "torch_dtype" in self.info:
            if self.info["torch_dtype"] == "fp32":
                torch_dtype = torch.float32
            elif self.info["torch_dtype"] == "bf16":
                torch_dtype = torch.bfloat16
            elif self.info["torch_dtype"] == "fp16":
                torch_dtype = torch.float16

        self.torch_dtype = torch_dtype

        model_kwargs = {"pretrained_model_name_or_path": self.pretrained_model_name,
                        "torch_dtype": torch_dtype,
                        "device_map": "auto",
                        "local_files_only": self.local_files_only,
                        "trust_remote_code": self.trust_remote_code}

        if self.quantization_config is not None:
            if isinstance(self.quantization_config, BitsAndBytesConfig):
                self.quantization_config.bnb_4bit_compute_dtype = TORCH_TYPE
            model_kwargs["quantization_config"] = self.quantization_config

        try:
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        except requests.exceptions.HTTPError:
            traceback.print_exc()
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first") from None
        except EnvironmentError:
            traceback.print_exc()
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first") from None

        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name,
                                                  trust_remote_code=self.trust_remote_code)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        if self.model_config.model_kwargs.get("use_generation_config", False):
            from transformers.generation import GenerationConfig
            self.generation_config = GenerationConfig.from_pretrained(
                self.pretrained_model_name,
                trust_remote_code=self.trust_remote_code)

        self.model = model
        self.tokenizer = tokenizer
        self.streamer = streamer

    def __del__(self):
        self.model = None
        self.tokenizer = None
        self.streamer = None

        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def info(self):
        return self.model_info


class HuggingFaceTransformersChat(HuggingFaceTransformers, ChatInterface):
    @torch.no_grad()
    def chat(self, messages, stream=False, options=None):
        options = options or dict()
        max_new_tokens = options.get("max_new_tokens", 512)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        prompt_tokens = len(model_inputs.input_ids[0])

        if not stream:
            generation_kwargs = dict(model_inputs, max_new_tokens=max_new_tokens)
            if self.model_config.model_kwargs.get("use_generation_config", False):
                generation_kwargs["generation_config"] = self.generation_config

            generated_ids = self.model.generate(**generation_kwargs)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            completion_tokens = len(generated_ids[0])

            return ChatCompletionResponse(**{"model": self.model_name,
                                             "content": content,
                                             "finish_reason": "stop" if completion_tokens < max_new_tokens else "length",

                                             "completion_tokens": completion_tokens,
                                             "prompt_tokens": prompt_tokens,
                                             "total_tokens": prompt_tokens+completion_tokens})
        else:
            def generator():
                generation_kwargs = dict(model_inputs, streamer=self.streamer, max_new_tokens=max_new_tokens)
                if self.model_config.model_kwargs.get("use_generation_config", False):
                    generation_kwargs["generation_config"] = self.generation_config

                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                completion_tokens = 0
                for content in self.streamer:
                    completion_tokens += 1

                    if not content:
                        continue

                    yield ChatCompletionStreamResponse(**{"model": self.model_name,
                                                          "delta_content": content})

                yield ChatCompletionStreamResponseDone(**{"model": self.model_name,
                                                          "finish_reason": "stop" if completion_tokens < max_new_tokens else "length",

                                                          "prompt_tokens": prompt_tokens,
                                                          "completion_tokens": completion_tokens,
                                                          "total_tokens": prompt_tokens + completion_tokens})
                thread.join()
            return generator()


def run_test(model_name, stream=False, **kwargs):
    import inspect
    print("=" * 80)

    model = HuggingFaceTransformersChat(model_name, local_files_only=False, **kwargs)
    model.load()
    print(model.info)

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
    model_name = "Qwen/Qwen2-0.5B-Instruct"

    run_test(model_name, stream=False)
    run_test(model_name, stream=True)