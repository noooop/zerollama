

from zerollama.core.config.main import config_setup

config_setup()

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

import torch
import numpy as np
import zmq
import transformers
import accelerate
import optimum
import auto_gptq
import awq

print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("zmq:", zmq.__version__)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("auto-gptq:", auto_gptq.__version__)
print("autoawq:", awq.__version__)


@torch.no_grad()
def chat(model_name):
    print(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "GPTQ" in model_name or "AWQ" in model_name:
        torch_dtype = torch.float16
    else:
        torch_dtype = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto"
    )

    model = model.to(device)

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_length=512,
    )


if __name__ == '__main__':
    import traceback
    for model_name in ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",
                       "01-ai/Yi-6B-Chat-8bits"]:
        try:
            chat(model_name)
        except Exception:
            traceback.print_exc()


r"""
numpy: 1.26.4
torch: 2.2.2+cu121
zmq: 25.1.2
transformers: 4.38.2
accelerate: 0.29.2
auto-gptq: 0.7.1
autoawq: 0.2.4

Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8
Traceback (most recent call last):
  File "Y:\PycharmProjects\zerollama\test\debug\gptq_int8.py", line 72, in <module>
    chat(model_name)
  File "C:\Users\noooop\anaconda3\Lib\site-packages\torch\utils\_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "Y:\PycharmProjects\zerollama\test\debug\gptq_int8.py", line 61, in chat
    generated_ids = model.generate(
                    ^^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\torch\utils\_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\transformers\generation\utils.py", line 1592, in generate
    return self.sample(
           ^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\transformers\generation\utils.py", line 2734, in sample
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0

01-ai/Yi-6B-Chat-8bits
Traceback (most recent call last):
  File "Y:\PycharmProjects\zerollama\test\debug\gptq_int8.py", line 72, in <module>
    chat(model_name)
  File "C:\Users\noooop\anaconda3\Lib\site-packages\torch\utils\_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "Y:\PycharmProjects\zerollama\test\debug\gptq_int8.py", line 61, in chat
    generated_ids = model.generate(
                    ^^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\torch\utils\_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\transformers\generation\utils.py", line 1592, in generate
    return self.sample(
           ^^^^^^^^^^^^
  File "C:\Users\noooop\anaconda3\Lib\site-packages\transformers\generation\utils.py", line 2734, in sample
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
"""