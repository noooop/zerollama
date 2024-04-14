

import torch
from threading import Thread
from src.core.models.chat import ChatInterfaces
from src.core.config.main import config_setup


class Qwen1_5(ChatInterfaces):
    def __init__(self, model_name, device="cuda"):
        if model_name not in info_dict:
            raise KeyError(f"{model_name} not in Qwen1.5 model family.")

        self.device = device
        self.model_name = model_name
        self.info = info_dict[self.model_name]
        self.model = None
        self.tokenizer = None
        self.streamer = None
        self.eos_token_id = None

    def load(self):
        config_setup()
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

        torch_dtype = torch.float16 if self.info["quantization"] != "" else "auto"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.streamer = streamer
        self.eos_token_id = tokenizer.encode('<|im_end|>')

    @torch.no_grad()
    def chat(self, messages, options=None):
        options = options or dict()
        max_new_tokens = options.get("max_new_tokens", 128)

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

        result = {
            "response_text": response,
            "response_length": response_length,
            "prompt_length": prompt_length,
            "finish_reason": "stop" if response_length < max_new_tokens else "length"
        }

        return result

    @torch.no_grad()
    def stream_chat(self, messages, options=None):
        options = options or dict()
        max_new_tokens = options.get("max_new_tokens", 128)

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

    @property
    def model_info(self):
        return self.info


info_header = ["name", "family", "type", "size", "quantization", "bits"]
info = [
    # name                                   family     type    size      quantization(_, GPTQ, AWQ)     bits
    # original
    ["Qwen/Qwen1.5-0.5B-Chat",               "Qwen1.5", "Chat", "0.5B",   "",                            ""],
    ["Qwen/Qwen1.5-1.8B-Chat",               "Qwen1.5", "Chat", "1.8B",   "",                            ""],
    ["Qwen/Qwen1.5-4B-Chat",                 "Qwen1.5", "Chat", "4B",     "",                            ""],
    ["Qwen/Qwen1.5-7B-Chat",                 "Qwen1.5", "Chat", "7B",     "",                            ""],
    ["Qwen/Qwen1.5-14B-Chat",                "Qwen1.5", "Chat", "14B",    "",                            ""],
    ["Qwen/Qwen1.5-32B-Chat",                "Qwen1.5", "Chat", "32B",    "",                            ""],
    ["Qwen/Qwen1.5-72B-Chat",                "Qwen1.5", "Chat", "72B",    "",                            ""],

    # GPTQ-Int8
    ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",     "Qwen1.5", "Chat", "0.5B",   "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8",     "Qwen1.5", "Chat", "1.8B",   "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-4B-Chat-GPTQ-Int8",       "Qwen1.5", "Chat", "4B",     "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",       "Qwen1.5", "Chat", "7B",     "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",      "Qwen1.5", "Chat", "14B",    "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-32B-Chat-GPTQ-Int8",      "Qwen1.5", "Chat", "32B",    "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-72B-Chat-GPTQ-Int8",      "Qwen1.5", "Chat", "72B",    "GPTQ",                        "Int8"],

    # GPTQ-Int4
    ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",     "Qwen1.5", "Chat", "0.5B",   "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4",     "Qwen1.5", "Chat", "1.8B",   "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-4B-Chat-GPTQ-Int4",       "Qwen1.5", "Chat", "4B",     "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",       "Qwen1.5", "Chat", "7B",     "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-14B-Chat-GPTQ-Int4",      "Qwen1.5", "Chat", "14B",    "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",      "Qwen1.5", "Chat", "32B",    "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-72B-Chat-GPTQ-Int4",      "Qwen1.5", "Chat", "72B",    "GPTQ",                        "Int4"],

    # AWQ
    ["Qwen/Qwen1.5-0.5B-Chat-AWQ",           "Qwen1.5", "Chat", "0.5B",   "AWQ",                         ""],
    ["Qwen/Qwen1.5-1.8B-Chat-AWQ",           "Qwen1.5", "Chat", "1.8B",   "AWQ",                         ""],
    ["Qwen/Qwen1.5-4B-Chat-AWQ",             "Qwen1.5", "Chat", "4B",     "AWQ",                         ""],
    ["Qwen/Qwen1.5-7B-Chat-AWQ",             "Qwen1.5", "Chat", "7B",     "AWQ",                         ""],
    ["Qwen/Qwen1.5-14B-Chat-AWQ",            "Qwen1.5", "Chat", "14B",    "AWQ",                         ""],
    ["Qwen/Qwen1.5-32B-Chat-AWQ",            "Qwen1.5", "Chat", "32B",    "AWQ",                         ""],
    ["Qwen/Qwen1.5-72B-Chat-AWQ",            "Qwen1.5", "Chat", "72B",    "AWQ",                         ""],
]
info_dict = {x[0]: {k: v for k, v in zip(info_header, x)} for x in info}


if __name__ == '__main__':
    for model_name in ["Qwen/Qwen1.5-0.5B-Chat",
                      #"Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",
                       "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",
                       "Qwen/Qwen1.5-0.5B-Chat-AWQ"]:
        print("\n\n")
        print("=" * 80)
        qwen = Qwen1_5(model_name)
        qwen.load()
        print(qwen.model_info)

        prompt = "给我介绍一下大型语言模型。"

        messages = [
            {"role": "user", "content": prompt}
        ]

        for response in qwen.stream_chat(messages):
            print(response, end="")