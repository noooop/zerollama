
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HOME'] = 'D:/.cache/huggingface/'

import warnings
warnings.filterwarnings("ignore")


import torch
from threading import Thread


class Qwen(object):
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.streamer = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.streamer = streamer

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
            max_new_tokens=max_new_tokens
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

        generation_kwargs = dict(model_inputs, streamer=self.streamer, max_new_tokens=max_new_tokens)

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for count, new_text in enumerate(self.streamer):
            result = {
                'count': count+1,
                "response_text": new_text,
                "done": False,
            }
            yield result


if __name__ == '__main__':
    qwen = Qwen("Qwen/Qwen1.5-0.5B-Chat")
    qwen.load()

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    for response in qwen.stream_chat(messages):
        print(response)