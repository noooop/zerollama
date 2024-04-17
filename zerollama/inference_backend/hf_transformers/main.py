

import torch
from threading import Thread
from zerollama.core.config.main import config_setup
from zerollama.core.models.chat import ChatInterfaces


class HuggingFaceTransformers(object):
    def __init__(self, model_name, info_dict, local_files_only=True, device="cuda"):
        if model_name not in info_dict:
            raise KeyError(f"{model_name} not in Qwen1.5 model family.")

        self.device = device
        self.model_name = model_name
        self.info = info_dict[self.model_name]
        self.model = None
        self.tokenizer = None
        self.streamer = None
        self.eos_token_id = None
        self.local_files_only = local_files_only

    def _load(self):
        pass

    def load(self):
        config_setup()
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

        torch_dtype = torch.float16 if self.info["quantization"] != "" else "auto"

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                local_files_only=self.local_files_only
            )
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.streamer = streamer
        self._load()
        self.eos_token_id = tokenizer.encode('<|im_end|>')

    @property
    def model_info(self):
        return self.info


class HuggingFaceTransformersChat(HuggingFaceTransformers, ChatInterfaces):
    @torch.no_grad()
    def chat(self, messages, options=None):
        options = options or dict()
        max_new_tokens = options.get("max_new_tokens", 512)

        messages = [{"role": "system", "content": "你是一个有用的助手。"}] + messages

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        prompt_length = len(model_inputs.input_ids[0])

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        response_length = len(generated_ids[0])

        return response

    @torch.no_grad()
    def stream_chat(self, messages, options=None):
        options = options or dict()
        max_new_tokens = options.get("max_new_tokens", 512)

        messages = [{"role": "system", "content": "你是一个有用的助手。"}] + messages

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generation_kwargs = dict(model_inputs, streamer=self.streamer,
                                 max_new_tokens=max_new_tokens, eos_token_id=self.eos_token_id)

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in self.streamer:
            if not new_text:
                continue
            yield new_text



