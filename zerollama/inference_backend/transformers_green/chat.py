
import gc
import torch
import requests
from threading import Thread
from zerollama.core.config.main import config_setup
from zerollama.tasks.chat.interface import ChatInterface
from zerollama.tasks.chat.protocol import ChatCompletionResponse, ChatCompletionStreamResponse
from zerollama.tasks.chat.collection import get_model_config_by_name


class HuggingFaceTransformers(object):
    def __init__(self, model_name, local_files_only=True, device="cuda"):
        model_config = get_model_config_by_name(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.local_files_only = local_files_only
        self.trust_remote_code = self.model_config.model_kwargs.get("trust_remote_code", False)

        self.model = None
        self.tokenizer = None
        self.streamer = None
        self.batch_size = 1

    def load(self):
        config = config_setup()

        if config.use_modelscope:
            from modelscope import AutoModelForCausalLM, AutoTokenizer
            from transformers import TextIteratorStreamer
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

        torch_dtype = "auto"
        if self.info["quantization"] != "":
            torch_dtype = torch.float16

        if "torch_dtype" in self.info:
            if self.info["torch_dtype"] == "fp32":
                torch_dtype = torch.float32
            elif self.info["torch_dtype"] == "bf16":
                torch_dtype = torch.bfloat16
            elif self.info["torch_dtype"] == "fp16":
                torch_dtype = torch.float16

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code
            )
        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first") from None
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first") from None

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.streamer = streamer

    def __del__(self):
        self.model = None
        self.tokenizer = None
        self.streamer = None

        gc.collect()
        torch.cuda.empty_cache()

    @property
    def info(self):
        return self.model_info


class HuggingFaceTransformersChat(HuggingFaceTransformers, ChatInterface):
    @torch.no_grad()
    def chat(self, messages, options=None):
        options = options or dict()
        max_new_tokens = options.get("max_new_tokens", 512)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        prompt_length = len(model_inputs.input_ids[0])

        content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        response_length = len(generated_ids[0])

        return ChatCompletionResponse(**{"model": self.model_name,
                                         "prompt_length": prompt_length,
                                         "content": content,
                                         "response_length": response_length,
                                         "finish_reason": "stop" if response_length < max_new_tokens else "length"})

    @torch.no_grad()
    def stream_chat(self, messages, options=None):
        options = options or dict()
        max_new_tokens = options.get("max_new_tokens", 512)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        prompt_length = len(model_inputs.input_ids[0])

        generation_kwargs = dict(model_inputs, streamer=self.streamer, max_new_tokens=max_new_tokens)

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        response_length = 0

        for content in self.streamer:
            if not content:
                continue
            response_length += 1
            yield ChatCompletionStreamResponse(**{"model": self.model_name,
                                                  "prompt_length": prompt_length,
                                                  "response_length": response_length,
                                                  "content": content,
                                                  "done": False})

        yield ChatCompletionStreamResponse(**{"model": self.model_name,
                                              "prompt_length": prompt_length,
                                              "response_length": response_length,
                                              "finish_reason": "stop" if response_length < max_new_tokens else "length",
                                              "done": True})


def run_test(model_name, stream=False):
    print("=" * 80)

    model = HuggingFaceTransformersChat(model_name, local_files_only=False)
    model.load()
    print(model.info)

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    if stream:
        for response in model.stream_chat(messages):
            if not response.done:
                print(response.content, end="", flush=True)
            else:
                print()
                print("response_length:", response.response_length)
    else:
        response = model.chat(messages)
        print(response.content)
        print("response_length:", response.response_length)
