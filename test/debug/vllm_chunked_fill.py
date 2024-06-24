
import os
from zerollama.core.config.main import config_setup
config_setup()

os.environ["VLLM_USE_MODELSCOPE"] = "False"

import traceback
from vllm.inputs import TokensPrompt
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams

max_num_batched_tokens = 64
max_num_seqs = 32


def decoding_latency(model_name, n, **kwargs):
    print(model_name)
    engine_args = EngineArgs(model=model_name,
                             disable_log_stats=True,
                             device="cuda",
                             max_model_len=1024,
                             enable_chunked_prefill=True,
                             max_num_batched_tokens=max_num_batched_tokens,
                             max_num_seqs=max_num_seqs,
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

    sampling_params = SamplingParams(ignore_eos=True, max_tokens=100)

    for i in range(n):
        engine.add_request(str(request_id), inputs, sampling_params)
        request_id += 1

    n_step = 0
    while engine.has_unfinished_requests():
        try:
            n_step += 1
            engine.step()
        except Exception:
            print("Exception at n_step:", n_step)
            traceback.print_exc()
            break


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor

    import torch
    import numpy as np
    import transformers
    import vllm

    print("numpy:", np.__version__)
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("vllm:", vllm.__version__)

    gptq_info = [
        # GPTQ-Int4
        ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",     "0.5B",   "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4",     "1.8B",   "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-4B-Chat-GPTQ-Int4",       "4B",     "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",       "7B",     "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-14B-Chat-GPTQ-Int4",      "14B",    "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",      "32B",    "GPTQ",                        "4bits"],
    ]

    for n in [128]:
        for model_name, *_ in gptq_info:
            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(decoding_latency, model_name, n)
                f.result()


"""
numpy: 1.26.4
torch: 2.3.0+cu121
transformers: 4.40.2
vllm: 0.5.0.post1
Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4 ok
Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4 ok
Qwen/Qwen1.5-4B-Chat-GPTQ-Int4

Exception at n_step: 325

Traceback (most recent call last):
  File "/media/noooop/cache/share/PycharmProjects/zerollama-vllm/test/debug/vllm_chunked_fill.py", line 56, in decoding_latency
    engine.step()
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/engine/llm_engine.py", line 776, in step
    output = self.model_executor.execute_model(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/executor/gpu_executor.py", line 91, in execute_model
    output = self.driver_worker.execute_model(execute_model_req)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/worker/worker.py", line 280, in execute_model
    output = self.model_runner.execute_model(seq_group_metadata_list,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/worker/model_runner.py", line 765, in execute_model
    output = self.model.sample(
             ^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/model_executor/models/qwen2.py", line 345, in sample
    next_tokens = self.sampler(logits, sampling_metadata)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/model_executor/layers/sampler.py", line 96, in forward
    sample_results, maybe_sampled_tokens_tensor = _sample(
                                                  ^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/model_executor/layers/sampler.py", line 655, in _sample
    return _sample_with_torch(
           ^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/model_executor/layers/sampler.py", line 544, in _sample_with_torch
    sample_results = _random_sample(seq_groups,
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/model_executor/layers/sampler.py", line 324, in _random_sample
    random_samples = random_samples.cpu()
                     ^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Qwen/Qwen1.5-7B-Chat-GPTQ-Int4
Exception at n_step: 325

Traceback (most recent call last):
  File "/media/noooop/cache/share/PycharmProjects/zerollama-vllm/test/debug/vllm_chunked_fill.py", line 56, in decoding_latency
    engine.step()
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/engine/llm_engine.py", line 776, in step
    output = self.model_executor.execute_model(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/executor/gpu_executor.py", line 91, in execute_model
    output = self.driver_worker.execute_model(execute_model_req)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/worker/worker.py", line 280, in execute_model
    output = self.model_runner.execute_model(seq_group_metadata_list,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/worker/model_runner.py", line 765, in execute_model
    output = self.model.sample(
             ^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/model_executor/models/qwen2.py", line 345, in sample
    next_tokens = self.sampler(logits, sampling_metadata)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/model_executor/layers/sampler.py", line 96, in forward
    sample_results, maybe_sampled_tokens_tensor = _sample(
                                                  ^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/model_executor/layers/sampler.py", line 655, in _sample
    return _sample_with_torch(
           ^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/model_executor/layers/sampler.py", line 544, in _sample_with_torch
    sample_results = _random_sample(seq_groups,
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noooop/anaconda3/envs/zerollama_v0.4/lib/python3.11/site-packages/vllm/model_executor/layers/sampler.py", line 324, in _random_sample
    random_samples = random_samples.cpu()
                     ^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Qwen/Qwen1.5-14B-Chat-GPTQ-Int4 ok
Qwen/Qwen1.5-32B-Chat-GPTQ-Int4 ok
"""