# Qwen (通义千问)
Alibaba Cloud's general-purpose AI models

## Qwen1.5

###  TL;DR
This time, we upgrade Qwen to Qwen1.5, the beta version of Qwen2. Similar to Qwen, it is still a decoder-only transformer model with SwiGLU activation, RoPE, multi-head attention. At this moment, we have achieved:
* 9 model sizes, including 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, and 110B dense models, and an MoE model of 14B with 2.7B activated;
* Significant model quality improvements in chat models;
* Strengthened multilingual capabilities in both base and chat models;
* All models support the context length of `32768` tokens;
* System prompts enabled for all models, which means roleplay is possible.
* No need of `trust_remote_code` anymore.

### News
* 2024.04.25: Released [110B model](https://qwenlm.github.io/blog/qwen1.5-110b/): 
* 2024.04.02: Released [32B model](https://qwenlm.github.io/zh/blog/qwen1.5-32b/): 
* 2024.03.28: Released [MoE model](https://qwenlm.github.io/zh/blog/qwen-moe/): Qwen1.5-MoE-A2.7B!
* 2024.02.05: Released [Qwen1.5 series](https://qwenlm.github.io/zh/blog/qwen1.5/):(0.5B, 1.8B, 4B, 7B, 14B and 72B).

### License Agreement
Check the license of each model inside its HF repo. It is NOT necessary for you to submit a request for commercial usage.


### Todo
GPTQ-Int8 模型报错, 感觉是GPTQ放弃Int8模型了
[Inference bug of the MoE GPTQ models](https://github.com/huggingface/transformers/issues/30515)


### Reference
[GITHUB](https://github.com/QwenLM/Qwen1.5)

[Hugging Face](https://huggingface.co/Qwen)

[ModelScope](https://modelscope.cn/organization/qwen)

[Blog](https://qwenlm.github.io/)

[Documentation](https://qwen.readthedocs.io/)
