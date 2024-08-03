# InternLM  (书生·浦语)
 InternLM is mainly developed by Shanghai AI Laboratory(上海人工智能实验室). 

## InternLM2-Reward
### TL;DR
InternLM2-Reward is a series of reward models, trained on 2.4 million preference samples, available in 1.8B, 7B, and 20B sizes. These model were applied to the PPO training process of our chat models. See [model cards](./model_cards/internlm2_reward.md) for more details.


### News
- \[2024.07.19\] We release the InternLM2-Reward series of reward models in 1.8B, 7B and 20B sizes. See [model zoo below](#model-zoo) for download or [model cards](./model_cards/internlm2_reward.md) for more details.

## InternLM2.5
### TL;DR

InternLM2.5 has open-sourced a 7 billion parameter base model and a chat model tailored for practical scenarios. The model has the following characteristics:

- **Outstanding reasoning capability**: State-of-the-art performance on Math reasoning, surpassing models like Llama3 and Gemma2-9B.

- **1M Context window**: Nearly perfect at finding needles in the haystack with 1M-long context, with leading performance on long-context tasks like LongBench. Try it with [LMDeploy](https://github.com/InternLM/InternLM/blob/main/chat/lmdeploy.md) for 1M-context inference.

- **Stronger tool use**: InternLM2.5 supports gathering information from more than 100 web pages, corresponding implementation will be released in [Lagent](https://github.com/InternLM/lagent/tree/main) soon. InternLM2.5 has better tool utilization-related capabilities in instruction following, tool selection and reflection. See [examples](https://github.com/InternLM/InternLM/blob/main/agent/lagent.md).

### News
- [2024.07.03] We release InternLM2.5-7B, InternLM2.5-7B-Chat and InternLM2.5-7B-Chat-1M. See [model zoo below](#model-zoo) for download or [model cards](./model_cards/) for more details.

- [2024.03.26] We release InternLM2 technical report. See [arXiv](https://arxiv.org/abs/2403.17297) for details.

- [2024.01.31] We release InternLM2-1.8B, along with the associated chat model. They provide a cheaper deployment option while maintaining leading performance.

- [2024.01.23] We release InternLM2-Math-7B and InternLM2-Math-20B with pretraining and SFT checkpoints. They surpass ChatGPT with small sizes. See [InternLM-Math](https://github.com/InternLM/internlm-math) for details and download.

- [2024.01.17] We release InternLM2-7B and InternLM2-20B and their corresponding chat models with stronger capabilities in all dimensions. See [model zoo below](#model-zoo) for download or [model cards](./model_cards/) for more details.

- [2023.12.13] InternLM-7B-Chat and InternLM-20B-Chat checkpoints are updated. With an improved finetuning strategy, the new chat models can generate higher quality responses with greater stylistic diversity.

- [2023.09.20] InternLM-20B is released with base and chat versions.

### License Agreement
The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow **free** commercial usage. To apply for a commercial license, please fill in the [application form (English)](https://wj.qq.com/s2/12727483/5dba/)/[申请表（中文）](https://wj.qq.com/s2/12725412/f7c1/). For other questions or collaborations, please contact <internlm@pjlab.org.cn>.

本仓库的代码依照 Apache-2.0 协议开源。模型权重对学术研究完全开放，也可申请免费的商业使用授权（[申请表](https://wj.qq.com/s2/12725412/f7c1/)）。其他问题与合作请联系 <internlm@pjlab.org.cn>。

### Reference
[GITHUB](https://github.com/InternLM/)

[huggingface](https://huggingface.co/internlm/)