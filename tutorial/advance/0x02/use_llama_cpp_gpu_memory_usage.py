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


def gpu_memory_usage(model_name, n_ctx):
    repo_id, filename = model_name.split("+")
    print(repo_id, filename, n_ctx)
    model = Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        verbose=False,

        n_gpu_layers=-1,
        n_ctx=n_ctx,
    )
    return True


if __name__ == '__main__':
    import traceback
    from concurrent.futures import ProcessPoolExecutor

    from zerollama.utils.logging import sys_logging
    from pathlib import Path

    sys_logging(Path("./llama_cpp/"))

    repo_id = [
        "Qwen/Qwen1.5-0.5B-Chat-GGUF",
        "Qwen/Qwen1.5-1.8B-Chat-GGUF",
        "Qwen/Qwen1.5-4B-Chat-GGUF",
        "Qwen/Qwen1.5-7B-Chat-GGUF",
        "Qwen/Qwen1.5-14B-Chat-GGUF",
        "Qwen/Qwen1.5-32B-Chat-GGUF",
    ]

    base_n_ctx = 10000

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

    Path("./llama_cpp/").mkdir(exist_ok=True)

    def test_oom(model_name, hight):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(gpu_memory_usage, model_name, hight)
            f.result()

    for r in repo_id:
        low = 0
        hight = base_n_ctx
        test_low = False

        for f in filename:
            model_name = f"{r}+{f}"

            while True:
                try:
                    test_oom(model_name, hight)
                except Exception:
                    break
                else:
                    test_low = False
                    low = hight
                    hight *= 2

            if test_low:
                while True:
                    try:
                        test_oom(model_name, low)
                    except Exception:
                        low //= 2
                    else:
                        break

            while (hight - low) > 32:
                mid = (low + hight) // 2
                try:
                    test_oom(model_name, mid)
                except Exception:
                    hight = mid
                else:
                    low = mid

            test_low = True
            print(model_name, low)
