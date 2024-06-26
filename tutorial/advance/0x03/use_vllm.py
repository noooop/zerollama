
from zerollama.core.config.main import config_setup
config_setup()

import time
import pickle
import random
from vllm.inputs import TokensPrompt
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
import numpy as np
import gc


def prefill_first_token_latency(model_name, seq_len, index, **kwargs):
    print(model_name)
    engine_args = EngineArgs(model=model_name, disable_log_stats=True, device="cuda", max_model_len=1024, **kwargs)
    engine = LLMEngine.from_engine_args(engine_args)

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    tokenizer = engine.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_tokens = tokenizer.encode(prompt)
    request_id = 0

    time_list = []
    for m in range(1, seq_len+1):
        if m < len(prompt_tokens):
            tokens = prompt_tokens[-m:]
        else:
            tokens = list(prompt_tokens)

        inputs = TokensPrompt(prompt_token_ids=tokens)

        sampling_params = SamplingParams(ignore_eos=True, max_tokens=1)

        try:
            tt = []
            for i in range(5):
                t = time.time()
                engine.add_request(str(request_id), inputs, sampling_params)
                request_id += 1

                n_step = 0
                while engine.has_unfinished_requests():
                    n_step += 1
                    engine.step()

                tt.append((time.time() - t, n_step))
        except Exception as e:
            raise e
            # traceback.print_exc()

        if m >= len(prompt_tokens):
            prompt_tokens = prompt_tokens[:20] + [random.randint(10000, 140000)] + prompt_tokens[20:]

        #print(m, len(tokens), f"{np.median(tt) * 1000:0.2f}")
        time_list.append((m, tt))

    pickle.dump(time_list, open(f"./vllm/{model_name.replace('/', '_')}-{index}.pkl", "wb"))

    del engine
    gc.collect()


def decoding_latency(model_name, idx, n, **kwargs):
    print(model_name)
    engine_args = EngineArgs(model=model_name,
                             disable_log_stats=True,
                             device="cuda",
                             max_model_len=1024,
                             **kwargs)
    engine = LLMEngine.from_engine_args(engine_args)

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    tokenizer = engine.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_tokens = tokenizer.encode(prompt)
    request_id = 0

    inputs = TokensPrompt(prompt_token_ids=prompt_tokens)

    n_prompt = len(prompt_tokens)

    data = []
    for index, max_len in enumerate([100, 1000, 1000, 1000]):
        time_list = [(time.time(), 0)]

        sampling_params = SamplingParams(ignore_eos=True, max_tokens=max_len)

        for i in range(n):
            engine.add_request(str(request_id), inputs, sampling_params)
            request_id += 1
        try:
            n_step = 0
            while engine.has_unfinished_requests():
                n_step += 1
                request_outputs = engine.step()

                time_list.append((time.time(), len(request_outputs)))
        except Exception:
            traceback.print_exc()

        y = [(time_list[i + 1][0] - time_list[i][0]) * 1000 for i in range(len(time_list) - 1)]
        x = list(range(n_prompt, len(y) + n_prompt))

        if index > 0:
            z = np.polyfit(x[100:], y[100:], 1)
            print(f"{model_name} {n_prompt} {n_step} {y[0]:0.2f} {1 / z[0]:0.0f} {z[1]:0.2f}")

        data.append(time_list)

    Path(f"./vllm/{n}/").mkdir(exist_ok=True, parents=True)

    pickle.dump(data, open(f"./vllm/{n}/{model_name.replace('/', '_')}-{idx}.pkl", "wb"))

    del engine
    gc.collect()


if __name__ == '__main__':
    import traceback
    from zerollama.utils.logging import sys_logging
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor

    bf16_info = [
        # name                                   size      quantization(_, GPTQ, AWQ)     bits
        # original
        ["Qwen/Qwen1.5-0.5B-Chat",               "0.5B",   "",                            ""],
        ["Qwen/Qwen1.5-1.8B-Chat",               "1.8B",   "",                            ""],
        ["Qwen/Qwen1.5-4B-Chat",                 "4B",     "",                            ""],
        ["Qwen/Qwen1.5-7B-Chat",                 "7B",     "",                            ""],
        ["Qwen/Qwen1.5-14B-Chat",                "14B",    "",                            ""],
        ["Qwen/Qwen1.5-32B-Chat",                "32B",    "",                            ""],
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

    sys_logging(Path("./vllm/"))

    test_prefill_first_token_latency = False
    test_decoding_latency = True

    if test_prefill_first_token_latency:
        for model_name, *_ in bf16_info:
            for index, kwargs in enumerate([{},
                                            {"quantization": "fp8"}]):
                try:
                    with ProcessPoolExecutor(1) as executor:
                        f = executor.submit(prefill_first_token_latency, model_name, 500, index, **kwargs)
                        f.result()
                except Exception as e:
                    traceback.print_exc()
                    print(model_name, "Exception")

        for model_name, *_ in gptq_info:
            for index, kwargs in enumerate([{}]):
                try:
                    with ProcessPoolExecutor(1) as executor:
                        f = executor.submit(prefill_first_token_latency, model_name, 500, index, **kwargs)
                        f.result()
                except Exception as e:
                    traceback.print_exc()
                    print(model_name, "Exception")

        for model_name, *_ in awq_info:
            for index, kwargs in enumerate([{}]):
                try:
                    with ProcessPoolExecutor(1) as executor:
                        f = executor.submit(prefill_first_token_latency, model_name, 500, index, **kwargs)
                        f.result()
                except Exception as e:
                    traceback.print_exc()
                    print(model_name, "Exception")

    if test_decoding_latency:
        for n in [1, 2, 4, 8, 16, 32, 64, 128]:
            for model_name, *_ in bf16_info:
                for index, kwargs in enumerate([{},
                                                {"quantization": "fp8"}]):
                    try:
                        with ProcessPoolExecutor(1) as executor:
                            f = executor.submit(decoding_latency, model_name, index, n, **kwargs)
                            f.result()
                    except Exception as e:
                        traceback.print_exc()
                        print(model_name, "Exception")

            for model_name, *_ in gptq_info:
                for index, kwargs in enumerate([{}]):
                    try:
                        with ProcessPoolExecutor(1) as executor:
                            f = executor.submit(decoding_latency, model_name, index, n, **kwargs)
                            f.result()
                    except Exception as e:
                        traceback.print_exc()
                        print(model_name, "Exception")

            for model_name, *_ in awq_info:
                for index, kwargs in enumerate([{}]):
                    try:
                        with ProcessPoolExecutor(1) as executor:
                            f = executor.submit(decoding_latency, model_name, index, n, **kwargs)
                            f.result()
                    except Exception as e:
                        traceback.print_exc()
                        print(model_name, "Exception")