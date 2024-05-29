import gc
import time
import pickle
import torch
import numpy as np
import random
from queue import Queue
from threading import Thread
from zerollama.core.config.main import config_setup

config_setup()

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from transformers.generation.streamers import BaseStreamer

from awq import AutoAWQForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import BitsAndBytesConfig

B = 1024 * 1024 * 1024
device = "cuda:0"


class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = True):
        self.skip_prompt = skip_prompt

        self.token_queue = Queue()
        self.next_tokens_are_prompt = True
        self.stop_signal = None

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.token_queue.put(value)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get()
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


@torch.no_grad()
def decoding_latency(model_name, idx, **kwargs):
    print(model_name, idx)
    print(torch.cuda.memory_allocated() / 1024 ** 3)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "GPTQ" in model_name:
        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            device=device,
            **kwargs
        )
    elif "AWQ" in model_name:
        model = AutoAWQForCausalLM.from_quantized(
            model_name,
            device=device,
            **kwargs
        )
        print(model.quant_config)
    else:
        if "quantization_config" in kwargs:
            torch_dtype = torch.float16
        else:
            torch_dtype = "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            **kwargs
        )

    streamer = TokenStreamer(skip_prompt=True)

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

    def generate(**kwargs):
        streamer = kwargs["streamer"]
        try:
            model.generate(**kwargs)
        except Exception:
            streamer.end()

    data = []
    for index, max_len in enumerate([100, 10000, 10000, 10000]):
        time_list = [time.time()]

        generation_kwargs = dict(model_inputs,
                                 streamer=streamer,
                                 max_length=max_len,
                                 pad_token_id=200000,
                                 eos_token_id=200000)

        thread = Thread(target=generate, kwargs=generation_kwargs)
        thread.start()

        completion_tokens = []
        for token in streamer:
            completion_tokens.append(token)
            time_list.append(time.time())

        n_prompt = model_inputs.input_ids.shape[-1]

        y = [(time_list[i + 1] - time_list[i]) * 1000 for i in range(len(time_list) - 1)]
        x = list(range(n_prompt, len(y) + n_prompt))

        if index > 0:
            z = np.polyfit(x[100:], y[100:], 1)
            print(f"{model_name} {n_prompt} {len(completion_tokens)} {y[0]:0.2f} {1 / z[0]:0.0f} {z[1]:0.2f}")

        data.append(time_list)
        thread.join()

    tokenizer = None
    model = None

    gc.collect()
    torch.cuda.empty_cache()

    pickle.dump(data, open(f"./hf/{model_name.replace('/', '_')}-{idx}.pkl", "wb"))


@torch.no_grad()
def gpu_memory_usage(model_name, idx, **kwargs):
    print(model_name, idx)
    print(torch.cuda.memory_allocated() / 1024 ** 3)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "GPTQ" in model_name:
        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            device=device,
            **kwargs
        )
    elif "AWQ" in model_name:
        model = AutoAWQForCausalLM.from_quantized(
            model_name,
            device=device,
            **kwargs
        )
        print(model.quant_config)
    else:
        if "quantization_config" in kwargs:
            torch_dtype = torch.float16
        else:
            torch_dtype = "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            **kwargs
        )

    streamer = TokenStreamer(skip_prompt=True)

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

    def generate(**kwargs):
        streamer = kwargs["streamer"]
        try:
            model.generate(**kwargs)
        except Exception:
            streamer.end()

    data = []
    for index, max_len in enumerate([100, 100000]):
        time_list = [time.time()]

        generation_kwargs = dict(model_inputs,
                                 streamer=streamer,
                                 max_length=max_len,
                                 pad_token_id=200000,
                                 eos_token_id=200000)

        thread = Thread(target=generate, kwargs=generation_kwargs)
        thread.start()

        completion_tokens = []
        for token in streamer:
            completion_tokens.append(token)
            time_list.append(time.time())

        n_prompt = model_inputs.input_ids.shape[-1]

        y = [(time_list[i + 1] - time_list[i]) * 1000 for i in range(len(time_list) - 1)]
        x = list(range(n_prompt, len(y) + n_prompt))

        if index > 0:
            z = np.polyfit(x[100:], y[100:], 1)
            print(f"{model_name} {n_prompt} {len(completion_tokens)} {n_prompt + len(completion_tokens)} {y[0]:0.2f} {1 / z[0]:0.0f} {z[1]:0.2f}")

        data.append(time_list)
        thread.join()

    tokenizer = None
    model = None

    gc.collect()
    torch.cuda.empty_cache()

    pickle.dump(data, open(f"./hf/{model_name.replace('/', '_')}-{idx}.pkl", "wb"))


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

    bf16_info = [
        # name                                   size      quantization(_, GPTQ, AWQ)     bits
        # original
        ["Qwen/Qwen1.5-0.5B-Chat",               "0.5B",   "",                            ""],
        ["Qwen/Qwen1.5-1.8B-Chat",               "1.8B",   "",                            ""],
        ["Qwen/Qwen1.5-4B-Chat",                 "4B",     "",                            ""],
        ["Qwen/Qwen1.5-7B-Chat",                 "7B",     "",                            ""],
        ["Qwen/Qwen1.5-4B-Chat",                 "14B",    "",                            ""],
        ["Qwen/Qwen1.5-7B-Chat",                 "21B",    "",                            ""],
    ]

    gptq_info = [
        # GPTQ-Int4
        ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",     "0.5B",   "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4",     "1.8B",   "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-4B-Chat-GPTQ-Int4",       "4B",     "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",       "7B",     "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-14B-Chat-GPTQ-Int4",      "14B",    "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",      "32B",    "GPTQ",                        "4bits"],
    ]

    awq_info = [
        # AWQ
        ["Qwen/Qwen1.5-0.5B-Chat-AWQ",           "0.5B",   "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-1.8B-Chat-AWQ",           "1.8B",   "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-4B-Chat-AWQ",             "4B",     "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-7B-Chat-AWQ",             "7B",     "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-14B-Chat-AWQ",            "14B",    "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-32B-Chat-AWQ",            "32B",    "AWQ",                         "4bits"],
    ]

    awq_kwargs = [
        {"max_seq_len": 11000,
          "batch_size": 1,
          "fuse_layers": False,
          "use_exllama": False,
          "use_exllama_v2": False},
        {"max_seq_len": 11000,
         "batch_size": 1,
         "fuse_layers": True,
         "use_exllama": False,
         "use_exllama_v2": False},

        {"max_seq_len": 11000,
          "batch_size": 1,
          "fuse_layers": False,
          "use_exllama": True,
          "use_exllama_v2": False},
        {"max_seq_len": 11000,
         "batch_size": 1,
         "fuse_layers": True,
         "use_exllama": True,
         "use_exllama_v2": False},

        {"max_seq_len": 11000,
          "batch_size": 1,
          "fuse_layers": False,
          "use_exllama": False,
          "use_exllama_v2": True},
        {"max_seq_len": 11000,
         "batch_size": 1,
         "fuse_layers": True,
         "use_exllama": False,
         "use_exllama_v2": True},
    ]

    awq_kwargs2 = [
        {"fuse_layers": False,
         "use_exllama": False,
         "use_exllama_v2": False},
        {"fuse_layers": True,
         "use_exllama": False,
         "use_exllama_v2": False},

        {"fuse_layers": False,
         "use_exllama": True,
         "use_exllama_v2": False},

        {"fuse_layers": True,
         "use_exllama": True,
         "use_exllama_v2": False},

        {"fuse_layers": False,
         "use_exllama": False,
         "use_exllama_v2": True},
        {"fuse_layers": True,
         "use_exllama": False,
         "use_exllama_v2": True},
    ]

    test_gpu_memory_usage = False
    test_decoding_latency = True

    if test_gpu_memory_usage:
        print("gpu memory usage")
        for model_name, *_ in bf16_info:
            for index, kwargs in enumerate([{},
                                            {"quantization_config": BitsAndBytesConfig(load_in_8bit=True,
                                                                                           bnb_4bit_compute_dtype=torch.bfloat16)} ,
                                             {"quantization_config": BitsAndBytesConfig(load_in_4bit=True,
                                                                                        bnb_4bit_compute_dtype=torch.bfloat16)}
                                            ]):
                try:
                    gpu_memory_usage(model_name, index, **kwargs)
                except Exception as e:
                    traceback.print_exc()
                    print(model_name, "Exception")

        for model_name, *_ in gptq_info:
            for index, kwargs in enumerate([{}, {"use_marlin": True}]):
                try:
                    gpu_memory_usage(model_name, index, **kwargs)
                except Exception as e:
                    print(model_name, kwargs, "Exception")

        for model_name, *_ in awq_info:
            for index, kwargs in enumerate(awq_kwargs):
                try:
                    gpu_memory_usage(model_name, index, **kwargs)
                except Exception as e:
                    traceback.print_exc()
                    print(model_name, kwargs, "Exception")

    if test_decoding_latency:
        print("Decoding Latency")
        for model_name, *_ in bf16_info:
            for index, kwargs in enumerate([{},
                                            {"quantization_config": BitsAndBytesConfig(load_in_8bit=True,
                                                                                       bnb_4bit_compute_dtype=torch.bfloat16)},
                                            {"quantization_config": BitsAndBytesConfig(load_in_4bit=True,
                                                                                       bnb_4bit_compute_dtype=torch.bfloat16)}
                                            ]):
                try:
                    decoding_latency(model_name, index, **kwargs)
                except Exception as e:
                    traceback.print_exc()
                    print(model_name, "Exception")

        for model_name, *_ in gptq_info:
            for index, kwargs in enumerate([{}, {"use_marlin": True}]):
                try:
                    decoding_latency(model_name, index, **kwargs)
                except Exception as e:
                    print(model_name, kwargs, "Exception")

        for model_name, *_ in awq_info:
            for index, kwargs in enumerate(awq_kwargs):
                try:
                    gpu_memory_usage(model_name, index, **kwargs)
                except Exception as e:
                    traceback.print_exc()
                    print(model_name, kwargs, "Exception")

        for model_name, *_ in awq_info:
            for index, kwargs in enumerate(awq_kwargs2):
                try:
                    gpu_memory_usage(model_name, index+10, **kwargs)
                except Exception as e:
                    traceback.print_exc()
                    print(model_name, kwargs, "Exception")






