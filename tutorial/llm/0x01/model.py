
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HOME'] = 'D:/.cache/'


import torch


class Qwen(object):
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.eos_token_id = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.encode('<|im_end|>')

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

        result = {
                "response_text": response,
                "response_length": response_length,
                "prompt_length": prompt_length,
                "finish_reason": "stop" if response_length < max_new_tokens else "length"
            }

        return result


if __name__ == '__main__':
    from pprint import pprint
    qwen = Qwen("Qwen/Qwen1.5-0.5B-Chat")
    qwen.load()

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    pprint(qwen.chat(messages))