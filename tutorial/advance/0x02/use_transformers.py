
import gc
import time
import torch
import numpy as np
import random

from zerollama.core.config.main import config_setup

config_setup()

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from awq import AutoAWQForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


B = 1024 * 1024 * 1024
device = "cuda:0"


@torch.no_grad()
def decoding_latency(model_name):
    print(model_name)
    print(torch.cuda.memory_allocated() / 1024 ** 3)

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
    for m in [2048 * i for i in range(1, 9)]:
        try:
            tt = []
            for i in range(3):
                t = time.time()
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_length=m,
                    pad_token_id=200000,
                    eos_token_id=200000
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
def prefill_first_token_latency(model_name):
    print(model_name)
    print(torch.cuda.memory_allocated() / B)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "GPTQ" in model_name:
        #kwargs = {"use_marlin": True}
        kwargs = {}
        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            device=device,
            **kwargs
        )

    elif "AWQ" in model_name:
        kwargs = {"max_seq_len": 501,
                  "batch_size": 1,
                  "fuse_layers": True,
                  "use_exllama": False,
                  "use_exllama_v2": True}
        model = AutoAWQForCausalLM.from_quantized(
            model_name,
            **kwargs
        )
        print(model.quant_config)
    else:
        kwargs = {}
        torch_dtype = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            **kwargs
        )
        model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters:", total_params)

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

    """
    ['<|im_start|>'] ['system'] ['\n'] ['You'] [' are'] [' a'] [' helpful'] [' assistant'] ['.'] ['<|im_end|>'] ['\n'] 
    0                1          2      3       4        5      6            7              8     9              10
    ['<|im_start|>'] ['user'] ['\n'] ['给我'] ['介绍一下'] ['大型'] ['语言'] ['模型'] ['。'] 
    11               12       13     14      15          16      17      18       19
    ['<|im_end|>'] ['\n'] ['<|im_start|>'] ['assistant'] ['\n']
    20             21     22                23           24
    """

    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask

    time_list = []
    for m in range(input_ids.shape[-1], 501):
        try:
            tt = []
            for i in range(10):
                t = time.time()
                if "GPTQ" in model_name:
                    generated_ids = model.generate(
                        max_length=m + 1,
                        pad_token_id=200000,
                        eos_token_id=200000,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                else:
                    generated_ids = model.generate(
                        input_ids,
                        max_length=m+1,
                        pad_token_id=200000,
                        eos_token_id=200000,
                    )
                tt.append(time.time()-t)

            input_ids = torch.tensor([input_ids[0, :20].tolist() +
                                      [random.randint(10000, 140000)] +
                                      input_ids[0, 20:].tolist()],
                                     dtype=input_ids.dtype, device=device)
            attention_mask = torch.ones(input_ids.shape, dtype=attention_mask.dtype, device=device)
            print(m, f"{np.median(tt)*1000:0.2f}")
            time_list.append((m, tt))
        except Exception as e:
            raise e
            #traceback.print_exc()

    tokenizer = None
    model = None

    gc.collect()
    torch.cuda.empty_cache()
    return time_list


if __name__ == '__main__':
    import traceback
    from zerollama.utils.logging import sys_logging

    sys_logging()

    info = [
        # name                                   size      quantization(_, GPTQ, AWQ)     bits
        # original
        ["Qwen/Qwen1.5-0.5B-Chat",               "0.5B",   "",                            ""],
        ["Qwen/Qwen1.5-1.8B-Chat",               "1.8B",   "",                            ""],
        ["Qwen/Qwen1.5-4B-Chat",                 "4B",     "",                            ""],
        ["Qwen/Qwen1.5-7B-Chat",                 "7B",     "",                            ""],
    ]

    info = [
        # GPTQ-Int4
        ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",     "0.5B",   "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4",     "1.8B",   "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-4B-Chat-GPTQ-Int4",       "4B",     "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",       "7B",     "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-14B-Chat-GPTQ-Int4",      "14B",    "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",      "32B",    "GPTQ",                        "4bits"],
    ]

    info = [
        # AWQ
        ["Qwen/Qwen1.5-0.5B-Chat-AWQ",           "0.5B",   "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-1.8B-Chat-AWQ",           "1.8B",   "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-4B-Chat-AWQ",             "4B",     "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-7B-Chat-AWQ",             "7B",     "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-14B-Chat-AWQ",            "14B",    "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-32B-Chat-AWQ",            "32B",    "AWQ",                         "4bits"],
    ]

    print("Prefill First Token Latency")
    for model_name, *_ in info:
        try:
            prefill_first_token_latency(model_name)
        except Exception as e:
            #raise e
            traceback.print_exc()

    #print("Decoding Latency(")
    #for model_name, *_ in info:
    #    decoding_latency(model_name)






