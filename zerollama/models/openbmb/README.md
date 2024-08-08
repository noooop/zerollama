# OpenBMB
OpenBMB开源社区由清华大学自然语言处理实验室和面壁智能共同支持发起。

## MiniCPM-2B

###  TL;DR
MiniCPM 是面壁智能与清华大学自然语言处理实验室共同开源的系列端侧大模型，主体语言模型 MiniCPM-2B 仅有 24亿（2.4B）的非词嵌入参数量, 总计2.7B参数量。
- 经过 SFT 后，MiniCPM-2B 在公开综合性评测集上与 Mistral-7B 表现相近（中文、数学、代码能力更优），整体性能超越 Llama2-13B、MPT-30B、Falcon-40B 等模型。
- 经过 DPO 后，MiniCPM-2B 在当前最接近用户体感的评测集 MTBench 上也超越了 Llama2-70B-Chat、Vicuna-33B、Mistral-7B-Instruct-v0.1、Zephyr-7B-alpha 等众多代表性开源大模型。
- 以 MiniCPM-2B 为基础构建端侧多模态大模型 MiniCPM-V 2.0，在多个测试基准中实现了 7B 以下模型的最佳性能，在 OpenCompass 榜单上超过了 Qwen-VL-Chat 9.6B、CogVLM-Chat 17.4B 和 Yi-VL 34B 等更大参数规模的模型。MiniCPM-V 2.0 还展现出领先的 OCR 能力，在场景文字识别能力上接近 Gemini Pro。
- 经过 Int4 量化后，MiniCPM 可在手机上进行部署推理，流式输出速度略高于人类说话速度。MiniCPM-V 也直接跑通了多模态大模型在手机上的部署。
- 一张1080/2080可高效参数微调，一张3090/4090可全参数微调，一台机器可持续训练 MiniCPM，二次开发成本较低。

### News
- 2024/04/11 开源[MiniCPM-V-2.0](https://huggingface.co/openbmb/MiniCPM-V-2.0)、[MiniCPM-2B-128k](https://huggingface.co/openbmb/MiniCPM-2B-128k)、[MiniCPM-MoE-8x2B](https://huggingface.co/openbmb/MiniCPM-MoE-8x2B)和[MiniCPM-1B](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16)！点击[这里](https://openbmb.vercel.app/?category=Chinese+Blog)查看技术博客。** 
- 2024/03/16 MiniCPM-2B 的30余个中间检查点开放了！[HuggingFace链接](https://huggingface.co/openbmb/MiniCPM-2B-history)
- 2024/02/08 我们更新了[llama-format的模型权重](#llamaformat)，方便大家更加快捷地使用我们的模型。
- 2024/02/01 初始发布。

### License 
本仓库中代码依照 Apache-2.0 协议开源

MiniCPM 模型权重的使用则需要遵循 “通用模型许可协议-来源说明-宣传限制-商业授权”。

MiniCPM 模型权重对学术研究完全开放。

如需将模型用于商业用途，请联系cpm@modelbest.cn来获取书面授权，在登记后亦允许免费商业使用。

This repository is released under the Apache-2.0 License.

The usage of MiniCPM model weights must strictly follow the General Model License (GML).

The models and weights of MiniCPM are completely free for academic research.

If you intend to utilize the model for commercial purposes, please reach out to cpm@modelbest.cn to obtain the certificate of authorization.

### Reference
[Homepage](https://www.openbmb.cn/home)

[MiniCPM 技术报告](https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a)

[Technical Report ](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20?pvs=4)

[GITHUB](https://github.com/OpenBMB/MiniCPM)

[Hugging Face](https://huggingface.co/openbmb)

## MiniCPM-S-1B

###  TL;DR
The utilization of activation sparsity, namely the existence of considerable weakly-contributed elements among activation outputs, is a promising method for inference acceleration of large language models (LLMs) ([Liu et al., 2023](https://proceedings.mlr.press/v202/liu23am/liu23am.pdf); [Song et al., 2023](https://arxiv.org/pdf/2312.12456.pdf)). Concretely, acceleration methods based on activation sparsity usually achieve higher inference speed by making wiser resource allocation and computation policies to avoid resource waste on these weakly-contributed parameters.

Adopting ReLU as the activation function is a straightforward method to achieve activation sparsity. However, most recent mainstream LLMs adopt activation functions without intrinsic sparsity (e.g., GELU and Swish). Some efforts ([Zhang et al., 2022](https://aclanthology.org/2022.findings-acl.71.pdf); [Mirzadeh et al., 2023](https://arxiv.org/pdf/2310.04564.pdf); [Zhang et al., 2024](https://arxiv.org/pdf/2402.03804.pdf)) introduce ReLU or its variants as the substitutive activation function to help non-ReLU LLMs achieve activation sparsity and inference acceleration, but few can concurrently obtain high sparsity and comparable task-specific performance.

In this work, we introduce a simple and effective sparsification method named "ProSparse" to push LLMs for higher activation sparsity while maintaining comparable performance. By applying ProSparse to Swish-activated LLaMA2-7B, LLaMA2-13B, and MiniCPM-1B, we obtain ReLU-activated models with high sparsity of 89.32%, 88.80%, and 87.89%, respectively, while their performance is comparable to the original version. These present the most sparsely activated models among open-source LLaMA versions and competitive end-size models, considerably surpassing ReluLLaMA-7B (66.98%) and ReluLLaMA-13B (71.56%). Further inference acceleration experiments demonstrate the practical speedup effects of higher sparsity on both [PowerInfer](https://arxiv.org/pdf/2312.12456.pdf) and our two sparse GPU [operators](https://github.com/Raincleared-Song/sparse_gpu_operator).

### News
- 2024/07/04 发布 MiniCPM-S-1B。

### License 
This repository is released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License.

The usage of MiniCPM model weights must strictly follow [the General Model License (GML)](https://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md).

The models and weights of MiniCPM are completely free for academic research.

If you intend to utilize the model for commercial purposes, please reach out to cpm@modelbest.cn to obtain the certificate of authorization.

### Reference
[Hugging Face](https://huggingface.co/openbmb/MiniCPM-S-1B-sft/)

Paper: [link](https://arxiv.org/pdf/2402.13516.pdf) (Note: `MiniCPM-S-1B` is denoted as `ProSparse-1B` in the paper.)

## MiniCPM-V

###  TL;DR
**MiniCPM-V**是面向图文理解的端侧多模态大模型系列。该系列模型接受图像和文本输入，并提供高质量的文本输出。自2024年2月以来，我们共发布了4个版本模型，旨在实现**领先的性能和高效的部署**，目前该系列最值得关注的模型包括：

- **MiniCPM-Llama3-V 2.5**：🔥🔥🔥 MiniCPM-V系列的最新、性能最佳模型。总参数量8B，多模态综合性能超越 GPT-4V-1106、Gemini Pro、Claude 3、Qwen-VL-Max 等商用闭源模型，OCR 能力及指令跟随能力进一步提升，并支持超过30种语言的多模态交互。通过系统使用模型量化、CPU、NPU、编译优化等高效推理技术，MiniCPM-Llama3-V 2.5 可以实现高效的终端设备部署。

- **MiniCPM-V 2.0**：MiniCPM-V系列的最轻量级模型。总参数量2B，多模态综合性能超越 Yi-VL 34B、CogVLM-Chat 17B、Qwen-VL-Chat 10B 等更大参数规模的模型，可接受 180 万像素的任意长宽比图像输入，实现了和 Gemini Pro 相近的场景文字识别能力以及和 GPT-4V 相匹的低幻觉率。

MiniCPM-V 2.6 is the latest and most capable model in the MiniCPM-V series. The model is built on SigLip-400M and Qwen2-7B with a total of 8B parameters. It exhibits a significant performance improvement over MiniCPM-Llama3-V 2.5, and introduces new features for multi-image and video understanding. Notable features of MiniCPM-V 2.6 include:


### News
* [2024.08.06] 🔥🔥🔥 We open-source MiniCPM-V 2.6, which outperforms GPT-4V on single image, multi-image and video understanding. It advances popular features of MiniCPM-Llama3-V 2.5, and can support real-time video understanding on iPad. Try it now!
* [2024.08.03] MiniCPM-Llama3-V 2.5 technical report is released! See here.
* [2024.07.19] MiniCPM-Llama3-V 2.5 supports vLLM now! See here.
* [2024.05.28] 🚀🚀🚀 MiniCPM-Llama3-V 2.5 now fully supports its feature in llama.cpp and ollama! Please pull the latest code of our provided forks (llama.cpp, ollama). GGUF models in various sizes are available here. MiniCPM-Llama3-V 2.5 series is not supported by the official repositories yet, and we are working hard to merge PRs. Please stay tuned!
* [2024.05.28] 💫 We now support LoRA fine-tuning for MiniCPM-Llama3-V 2.5, using only 2 V100 GPUs! See more statistics here.
* [2024.05.23] 我们添加了Phi-3-vision-128k-instruct与MiniCPM-Llama3-V 2.5的全面对比，包括基准测试评估和多语言能力 🌟📊🌍。点击[这里](./docs/compare_with_phi-3_vision.md)查看详细信息。
<!-- * [2024.05.22] 我们进一步提升了端侧推理速度！实现了 6-8 tokens/s 的流畅体验，欢迎试用！ -->
* [2024.05.20] 我们开源了 MiniCPM-Llama3-V 2.5，增强了 OCR 能力，支持 30 多种语言，并首次在端侧实现了 GPT-4V 级的多模态能力！我们提供了[高效推理](#手机端部署)和[简易微调](./finetune/readme.md)的支持，欢迎试用！
* [2024.04.23] 我们增加了对 [vLLM](#vllm) 的支持，欢迎体验！
* [2024.04.18] 我们在 HuggingFace Space 新增了 MiniCPM-V 2.0 的 [demo](https://huggingface.co/spaces/openbmb/MiniCPM-V-2)，欢迎体验！
* [2024.04.17] MiniCPM-V 2.0 现在支持用户部署本地 [WebUI Demo](#本地webui-demo部署) 了，欢迎试用!
* [2024.04.15] MiniCPM-V 2.0 现在可以通过 SWIFT 框架 [微调](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v-2最佳实践.md) 了，支持流式输出!
* [2024.04.12] 我们开源了 MiniCPM-V 2.0，该模型刷新了 OCRBench 开源模型最佳成绩，在场景文字识别能力上比肩 Gemini Pro，同时还在综合了 11 个主流多模态大模型评测基准的 <a href="https://rank.opencompass.org.cn/leaderboard-multimodal">OpenCompass</a> 榜单上超过了 Qwen-VL-Chat 10B、CogVLM-Chat 17B 和 Yi-VL 34B 等更大参数规模的模型！点击<a href="https://openbmb.vercel.app/minicpm-v-2">这里</a>查看 MiniCPM-V 2.0 技术博客。
* [2024.03.14] MiniCPM-V 现在支持 SWIFT 框架下的[微调](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/minicpm-v最佳实践.md)了，感谢 [Jintao](https://github.com/Jintao-Huang) 的贡献！
* [2024.03.01] MiniCPM-V 现在支持在 Mac 电脑上进行部署！
* [2024.02.01] 我们开源了 MiniCPM-V 和 OmniLMM-12B，分别可以支持高效的端侧部署和同规模领先的多模态能力！

### 
MiniCPM-V 依赖 timm

### License 

本仓库中代码依照 Apache-2.0 协议开源

本项目中模型权重的使用遵循 “[通用模型许可协议-来源说明-宣传限制-商业授权](https://github.com/OpenBMB/General-Model-License/blob/main/通用模型许可协议-来源说明-宣传限制-商业授权.md)”。

本项目中模型权重对学术研究完全开放。

如需将模型用于商业用途，请联系 cpm@modelbest.cn 来获取书面授权，登记后可以免费商业使用。

The code in this repo is released according to [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE)

The usage of MiniCPM-V's and OmniLMM's parameters is subject to "[General Model License Agreement - Source Notes - Publicity Restrictions - Commercial License](https://github.com/OpenBMB/General-Model-License/blob/main/通用模型许可协议-来源说明-宣传限制-商业授权.md)"

The parameters are fully open to academic research

Please contact cpm@modelbest.cn to obtain written authorization for commercial uses. Free commercial use is also allowed after registration.

### Reference
[Homepage](https://www.openbmb.cn/home)

[MiniCPM-V 技术报告](https://openbmb.vercel.app/minicpm-v-2)

[GITHUB](https://github.com/OpenBMB/MiniCPM-V/)

[Hugging Face](https://huggingface.co/openbmb)
