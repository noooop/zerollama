# THUDM
The Knowledge Engineering Group (KEG) & Data Mining (THUDM) at Tsinghua University.

智谱AI是由清华大学计算机系技术成果转化而来的公司

## GLM-4

### TL;DR
GLM-4-9B 是智谱 AI 推出的最新一代预训练模型 GLM-4 系列中的开源版本。 在语义、数学、推理、代码和知识等多方面的数据集测评中，
**GLM-4-9B** 及其人类偏好对齐的版本 **GLM-4-9B-Chat** 均表现出超越 Llama-3-8B 的卓越性能。除了能进行多轮对话，GLM-4-9B-Chat
还具备网页浏览、代码执行、自定义工具调用（Function Call）和长文本推理（支持最大 128K 上下文）等高级功能。本代模型增加了多语言支持，支持包括日语，韩语，德语在内的
26 种语言。我们还推出了支持 1M 上下文长度（约 200 万中文字符）的 **GLM-4-9B-Chat-1M** 模型和基于 GLM-4-9B 的多模态模型
GLM-4V-9B。**GLM-4V-9B** 具备 1120 * 1120 高分辨率下的中英双语多轮对话能力，在中英文综合能力、感知推理、文字识别、图表理解等多方面多模态评测中，GLM-4V-9B
表现出超越 GPT-4-turbo-2024-04-09、Gemini
1.0 Pro、Qwen-VL-Max 和 Claude 3 Opus 的卓越性能。

### News

- 2024/6/5 released GLM-4-9B、GLM-4-9B-Chat、GLM-4-9B-Chat-1M、GLM-4V-9B

### License 

+ GLM-4 模型的权重的使用则需要遵循 [模型协议](https://huggingface.co/THUDM/glm-4-9b/blob/main/LICENSE)。

+ 本开源仓库的代码则遵循 [Apache 2.0](LICENSE) 协议。

请您严格遵循开源协议。

+ The use of GLM-4 model weights must follow
  the [Model License](https://huggingface.co/THUDM/glm-4-9b/blob/main/LICENSE).

+ The code in this open source repository follows the [Apache 2.0](LICENSE) license.

Please strictly follow the open source license.


### Reference
[GITHUB](https://github.com/THUDM/GLM-4)

[Hugging Face](https://huggingface.co/THUDM/)


## CogVLM2

###  TL;DR
We launch a new generation of **CogVLM2** series of models and open source two models based
on [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). Compared with the previous
generation of CogVLM open source models, the CogVLM2 series of open source models have the following improvements:

1. Significant improvements in many benchmarks such as `TextVQA`, `DocVQA`.
2. Support **8K** content length.
3. Support image resolution up to **1344 * 1344**.
4. Provide an open source model version that supports both **Chinese and English**.

### News
- 🔥🔥 **News**: ``2024/5/24```: We have released the Int4 version model, which requires only 16GB of video memory for inference. Welcome to experience it!
- 🔥 **News**: ``2024/5/20``: We released the next generation model CogVLM2, which is based on llama3-8b and is equivalent (or better) to GPT-4V in most cases ! Welcome to download!

### 
CogVLM2 依赖 xformers

### License 
This model is released under the CogVLM2 [CogVLM2 LICENSE](MODEL_LICENSE). For models built with Meta Llama 3, please
also adhere to the [LLAMA3_LICENSE](https://llama.meta.com/llama3/license/).

该模型根据 [CogVLM2 LICENSE](MODEL_LICENSE) 许可证发布。对于使用了Meta Llama
3基座模型构建的模型，需要同时遵守 [LLAMA3_LICENSE](https://llama.meta.com/llama3/license/) 许可证。

### Reference
[GITHUB](https://github.com/THUDM/CogVLM2/)

[Hugging Face](https://huggingface.co/THUDM/)

