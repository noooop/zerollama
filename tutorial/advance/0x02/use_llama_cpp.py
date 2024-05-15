
import time
import random
from tqdm import trange
from zerollama.core.config.main import config_setup

import torch

config_setup()

import pickle
import numpy as np
from llama_cpp.llama_chat_format import format_chatml
from llama_cpp import Llama, llama_token_is_eog

B = 1024 * 1024 * 1024
device = "cuda:0"

n_ctx = 600
top_k = 40
top_p = 0.95
min_p = 0.05
typical_p = 1.0
temp = 0.2
repeat_penalty = 1.1
frequency_penalty = 0.0
presence_penalty = 0.0
tfs_z = 1.0
mirostat_mode = 0
mirostat_tau = 5.0
mirostat_eta = 0.1
logits_processor = None
grammar = None
penalize_nl = True


def sample(model, sample_idx):
    token = model.sample(
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        typical_p=typical_p,
        temp=temp,
        repeat_penalty=repeat_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        tfs_z=tfs_z,
        mirostat_mode=mirostat_mode,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
        logits_processor=logits_processor,
        grammar=grammar,
        penalize_nl=penalize_nl,
        idx=sample_idx,
    )
    return token


def prefill_first_token_latency(model_name):
    print(model_name)
    print(torch.cuda.memory_allocated() / B)

    repo_id, filename = model_name.split("+")

    model = Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        verbose=False,

        n_gpu_layers=-1,
        n_ctx=n_ctx + 1,
        n_batch=n_ctx + 1,
    )

    ma = torch.cuda.memory_allocated()
    print(model_name, f"{ma / B:0.2f}GB")

    prompt_tokens = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 104169,
                     109432, 101951, 102064, 104949, 1773, 151645, 198, 151644, 77091, 198]

    time_list = []
    for m in range(len(prompt_tokens), 501):
        tokens = list(prompt_tokens)
        n_prompt_tokens = len(prompt_tokens)
        sample_idx = n_prompt_tokens - 1

        try:
            tt = []
            for i in range(10):
                model.reset()
                t = time.time()
                model.eval(tokens)
                token = sample(model, sample_idx)
                tt.append(time.time() - t)
            print(m, n_prompt_tokens, f"{np.median(tt) * 1000:0.2f}")
            time_list.append((m, tt))
        except Exception as e:
            raise e

        prompt_tokens = prompt_tokens[:20] + [random.randint(10000, 140000)] + prompt_tokens[20:]

    return time_list


if __name__ == '__main__':
    from zerollama.utils.logging import sys_logging

    sys_logging()


    repo_id = [
 #       "Qwen/Qwen1.5-0.5B-Chat-GGUF",
  #      "Qwen/Qwen1.5-1.8B-Chat-GGUF",
        "Qwen/Qwen1.5-4B-Chat-GGUF",
        "Qwen/Qwen1.5-7B-Chat-GGUF",
 #       "Qwen/Qwen1.5-14B-Chat-GGUF",
#        "Qwen/Qwen1.5-32B-Chat-GGUF",
#        "Qwen/Qwen1.5-72B-Chat-GGUF",
#        "Qwen/Qwen1.5-110B-Chat-GGUF",
    ]
    filename = [
        "*q8_0.gguf",
        "*q6_k.gguf",
        "*q5_k_m.gguf",
        "*q5_0.gguf",
        "*q4_k_m.gguf",
        "*q4_0.gguf",
        "*q3_k_m.gguf",
        "*q2_k.gguf"
    ]

    for r in repo_id:
        for f in filename:
            model_name = f"{r}+{f}"
            prefill_first_token_latency(model_name)
