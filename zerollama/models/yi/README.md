# Yi
Yi 系列模型是 [01.AI](https://01.ai/) 从零训练的下一代开源大语言模型。

##  TL;DR
- Yi 系列模型是一个双语语言模型，在 3T 多语言语料库上训练而成，是全球最强大的大语言模型之一。Yi 系列模型在语言认知、常识推理、阅读理解等方面表现优异。例如，

- Yi-34B-Chat 模型在 AlpacaEval Leaderboard [排名第二](https://twitter.com/01AI_Yi/status/1745371506623103087?s=20)，**仅次于 GPT-4 Turbo**，超过了 GPT-4、Mixtral 和 Claude 等大语言模型（数据截止至 2024 年 1 月）。

- Yi-34B 模型在 [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)（预训练）与 C-Eval 基准测试中[荣登榜首](https://mp.weixin.qq.com/s/tLP-fjwYHcXVLqDcrXva2g)，**在中文和英文语言能力方面**均超过了其它开源模型，例如，Falcon-180B、Llama-70B 和 Claude（数据截止至 2023 年 11 月）。

- （致谢 Llama ）感谢 Transformer 和 Llama 开源社区，不仅简化了开发者从零开始构建大模型的工作，开发者还可以利用 Llama 生态中现有的工具、库和资源，提高开发效率。


# Yi-1.5

##  TL;DR

Yi-1.5 is an upgraded version of Yi. It is continuously pre-trained on Yi with a high-quality corpus of 500B tokens and fine-tuned on 3M diverse fine-tuning samples. 

Compared with Yi, Yi-1.5 delivers stronger performance in coding, math, reasoning, and instruction-following capability, while still maintaining excellent capabilities in language understanding, commonsense reasoning, and reading comprehension. 

Yi-1.5 comes in 3 model sizes: 34B, 9B, and 6B. For model details and benchmarks, see [Model Card](https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8).



## Todo
GPTQ-Int8 模型报错 同 qwen 1.5

## News
- 2024-05-13: The Yi-1.5 series models are open-sourced, further improving coding, math, reasoning, and instruction-following abilities. 

🎯 2024-03-16：发布并开源了 Yi-9B-200K 模型。

🎯 2024-03-08: 发布了 Yi 技术报告！

🔔 2024-03-07: 增强了 Yi-34B-200K 长文本记忆和检索能力。

Yi-34B-200K 的“大海捞针”能力增强了 10.5%, 从 89.3% 提升到了 99.8%。 在 5B tokens 的长文本数据集上，对模型进行继续预训练，模型性能达到预期目标。

🎯 2024-03-06: 发布并开源了 Yi-9B 模型。

Yi-9B 模型在 Mistral-7B、SOLAR-10.7B、Gemma-7B、DeepSeek-Coder-7B-Base-v1.5 等相近尺寸的模型中名列前茅，具有出色的代码能力、数学能力、常识推理能力以及阅读理解能力。

🎯 2024-01-23: 发布并开源了 Yi-VL-34B 和 Yi-VL-6B 多模态语言大模型。

Yi-VL-34B在 MMMU 和 CMMMU 最新的基准测试中荣登榜首（数据截止至 2024 年 1 月）。

🎯 2023-11-23: 发布并开源了六大 Chat 模型。

🔔 2023-11-23： Yi 系列模型社区许可协议更新至 2.1 版本。 

🔥 2023-11-08： Yi-34B-Chat 模型开始邀请测试。

🎯 2023-11-05： 发布并开源了 Yi-6B-200K 和 Yi-34B-200K Base 模型。

🎯 2023-11-02： 发布并开源了 Yi-6B-Base 和 Yi-34B-Base 模型。


## License Agreement

The source code in this repo is licensed under the Apache 2.0 license. The Yi series models are fully open for academic research and free for commercial use, with automatic permission granted upon application. All usage must adhere to the Yi Series Models Community License Agreement 2.1. For free commercial use, you only need to send an email to get official commercial permission.

本仓库中的源代码遵循 Apache 2.0 许可证。Yi 系列模型完全开放，你可以免费用于个人用途、学术研究和商业用途。如需商用，你仅需提交申请，即能立刻自动获取 Yi 系列模型商用许可，而无需等待官方审批。所有使用必须遵守《Yi系列模型社区许可协议 2.1》。


## Reference
[GITHUB](https://github.com/01-ai/Yi/)

[Hugging Face](https://huggingface.co/01-ai/)

[ModelScope](https://www.modelscope.cn/organization/01ai/)

[Yi Tech Report](https://arxiv.org/abs/2403.04652)

