# Qwen (通义千问)
Alibaba Cloud's general-purpose AI models

## Qwen1.5

###  TL;DR
This time, we upgrade Qwen to Qwen1.5, the beta version of Qwen2. Similar to Qwen, it is still a decoder-only transformer model with SwiGLU activation, RoPE, multi-head attention. At this moment, we have achieved:
* 8 model sizes: 0.5B, 1.8B, 4B, 7B, 14B, 32B, and 72B models, plus a 14B (A2.7B) MoE model;
* Significant model quality improvements in chat models;
* Strengthened multilingual capabilities in both base and chat models;
* All models support the context length of `32768` tokens;
* System prompts enabled for all models, which means roleplay is possible.
* No need of `trust_remote_code` anymore.

### News
* 2024.04.02: Released [32B model](https://qwenlm.github.io/zh/blog/qwen1.5-32b/): 
* 2024.03.28: Released [MoE model](https://qwenlm.github.io/zh/blog/qwen-moe/): Qwen1.5-MoE-A2.7B!
* 2024.02.05: Released [Qwen1.5 series](https://qwenlm.github.io/zh/blog/qwen1.5/):(0.5B, 1.8B, 4B, 7B, 14B and 72B).

### License Agreement
Check the license of each model inside its HF repo. It is NOT necessary for you to submit a request for commercial usage.


### Todo
```
GPTQ-Int8 模型报错
Windows
{'name': 'Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8', 'family': 'Qwen1.5', 'type': 'Chat', 'size': '0.5B', 'quantization': 'GPTQ', 'bits': 'Int8'}
Exception in thread Thread-2 (generate):
Traceback (most recent call last):
  File "C:\Users\noooop\anaconda3\Lib\threading.py", line 1045, in _bootstrap_inner
    self.run()
  File "C:\Users\noooop\anaconda3\Lib\threading.py", line 982, in run
    self._target(*self._args, **self._kwargs)
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

Linux
{'name': 'Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8', 'family': 'Qwen1.5', 'type': 'Chat', 'size': '0.5B', 'quantization': 'GPTQ', 'bits': 'Int8'}
Exception in thread Thread-2 (generate):
Traceback (most recent call last):
  File "/root/anaconda3/lib/python3.11/threading.py", line 1045, in _bootstrap_inner
    self.run()
  File "/root/anaconda3/lib/python3.11/threading.py", line 982, in run
    self._target(*self._args, **self._kwargs)
  File "/root/anaconda3/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/anaconda3/lib/python3.11/site-packages/transformers/generation/utils.py", line 1592, in generate
    return self.sample(
           ^^^^^^^^^^^^
  File "/root/anaconda3/lib/python3.11/site-packages/transformers/generation/utils.py", line 2734, in sample
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0

```

### Reference
[GITHUB](https://github.com/QwenLM/Qwen1.5)

[Hugging Face](https://huggingface.co/Qwen)

[ModelScope](https://modelscope.cn/organization/qwen)

[Blog](https://qwenlm.github.io/)

[Documentation](https://qwen.readthedocs.io/)
