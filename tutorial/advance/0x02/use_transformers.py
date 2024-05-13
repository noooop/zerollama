
import gc
import time
import torch
import numpy as np

from zerollama.core.config.main import config_setup

config_setup()

from transformers import AutoModelForCausalLM, AutoTokenizer


B = 1024 * 1024 * 1024
device="cuda"


@torch.no_grad()
def decoding(model_name, max_tokens):
    print(model_name)
    print(torch.cuda.memory_allocated() / 1024 ** 3)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    model = model.to(device)

    ma = torch.cuda.memory_allocated()
    print(model_name, f"{ma / B:0.2f}GB")

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

    print(model_inputs.input_ids.shape)

    X, Y = [], []
    for m in max_tokens:
        try:
            tt = []
            for i in range(5):
                t = time.time()
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_length=m,
                    eos_token_id=[]
                )
                tt.append(time.time()-t)

            ma = torch.cuda.memory_allocated()
            mma = torch.cuda.max_memory_allocated()
            mmc = torch.cuda.max_memory_cached()

            print(model_name, generated_ids.shape, f"{ma / B:0.2f}GB", f"{mma / B:0.2f}GB", f"{mmc / B:0.2f}GB")
            print(m, np.median(tt)*1000)
            X.append(m)
            Y.append(np.median(tt)*1000)

        except Exception:
            pass

    z = np.polyfit(X, Y, 1)

    tokenizer = None
    model = None

    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def prefill_turning_point(model_name):
    print(model_name)
    print(torch.cuda.memory_allocated() / 1024 ** 3)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    model = model.to(device)

    ma = torch.cuda.memory_allocated()
    print(model_name, f"{ma / B:0.2f}GB")

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

    input_ids = model_inputs.input_ids

    time_list = []
    for m in range(input_ids.shape[-1], 201):
        try:
            tt = []
            for i in range(10):
                t = time.time()
                generated_ids = model.generate(
                    input_ids,
                    max_length=m+1,
                    eos_token_id=[]
                )
                tt.append(time.time()-t)
            input_ids = generated_ids
            print(m, input_ids.shape, np.median(tt)*1000)
            time_list.append((m, tt))
        except Exception:
            pass

    tokenizer = None
    model = None

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    info = [
        # name                                   size      quantization(_, GPTQ, AWQ)     bits
        # original
        ["Qwen/Qwen1.5-0.5B-Chat", "0.5B", "", ""],
        ["Qwen/Qwen1.5-1.8B-Chat", "1.8B", "", ""],
        ["Qwen/Qwen1.5-4B-Chat", "4B", "", ""],
        ["Qwen/Qwen1.5-7B-Chat", "7B", "", ""],
        # ["Qwen/Qwen1.5-14B-Chat", "14B", "", ""]
    ]

    print("Decoding")
    for model_name, *_ in info:
        decoding(model_name, max_tokens=[128 * i for i in range(1, 11)])

    print("prefill turning point")
    for model_name, *_ in info:
        prefill_turning_point(model_name)




