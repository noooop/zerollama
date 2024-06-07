# Qwen (通义千问)
Alibaba Cloud's general-purpose AI models

## Qwen2

###  TL;DR
After months of efforts, we are pleased to announce the evolution from Qwen1.5 to Qwen2. This time, we bring to you:
* Pretrained and instruction-tuned models of 5 sizes, including Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B, Qwen2-57B-A14B, and Qwen2-72B;
* Having been trained on data in 27 additional languages besides English and Chinese;
* State-of-the-art performance in a large number of benchmark evaluations;
* Significantly improved performance in coding and mathematics;
* Extended context length support up to 128K tokens with Qwen2-7B-Instruct and Qwen2-72B-Instruct

### News
* 2024.04.25: Released [Qwen2 series](https://qwenlm.github.io/blog/qwen2/): (Qwen2-0.5B、Qwen2-1.5B、Qwen2-7B、Qwen2-57B-A14B以及Qwen2-72B)

### License Agreement
此次我们采用不同的模型许可。除了Qwen2-72B依旧使用此前的Qianwen License外，其余模型，包括Qwen2-0.5B、Qwen2-1.5B、Qwen2-7B以及Qwen2-57B-A14B在内，均采用Apache 2.0的许可。我们希望本次开放程度的提升能够加速Qwen2在全球各地的落地及商业应用。

This time, we change the licenses of our models to different ones. While Qwen2-72B as well as its instruction-tuned models still uses the original Qianwen License, all other models, including Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B, and Qwen2-57B-A14B, turn to adopt Apache 2.0! We believe that the enhanced openness of our models to the community can accelerate the applications and commercial usages of Qwen2 all around the world.

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
