# XVERSE
XVERSE是由深圳元象科技自主研发的支持多语言的大语言模型（Large Language Model）

XVERSE: A multilingual large language model developed by XVERSE Technology Inc.

##  TL;DR

- 模型家族包括 7B、13B、65B、MoE-A4.2B

- 模型结构：XVERSE-65B 使用主流 Decoder-only 的标准 Transformer 网络结构，支持 16K 的上下文长度（Context Length），能满足更长的多轮对话、知识问答与摘要等需求，模型应用场景更广泛。
- 训练数据：构建了 2.6 万亿 token 的高质量、多样化的数据对模型进行充分训练，包含中、英、俄、西等 40 多种语言，通过精细化设置不同类型数据的采样比例，使得中英两种语言表现优异，也能兼顾其他语言效果。
- 分词：基于 BPE（Byte-Pair Encoding）算法，使用上百 GB 语料训练了一个词表大小为 100,534 的分词器，能够同时支持多语言，而无需额外扩展词表。
- 训练框架：训练中采用 FlashAttention2 加速计算，3D 并行基础上采用虚拟流水线（virtual pipeline）技术，降低较长流水线和 16k 上下文窗口产生的过高气泡率，在千卡集群的峰值算力利用率达到业界前列。同时通过集群基础设施运营、资源调度、训练框架和调度平台协同等持续优化，打造出高稳定、低中断、强容错的训练系统，将每周有效训练率提升至 98.6%。



## News
- [2024/04/28] 发布 MoE 架构的 XVERSE-MoE-A4.2B-Chat 对话模型。
- [2024/04/02] 发布 MoE 架构的 XVERSE-MoE-A4.2B 底座模型，Chat 对齐模型将在后续发布。
- [2024/03/25] 发布XVERSE-65B-Chat GGUF、GPTQ量化模型，支持llama.cpp、vLLM在MacOS/Linux/Windows系统上推理XVERSE-65B-Chat模型。
- [2024/03/25] 发布XVERSE-13B-2-Chat GGUF、GPTQ量化模型，支持llama.cpp、vLLM在MacOS/Linux/Windows系统上推理XVERSE-13B-2-Chat模型。
- [2024/01/16] 发布长序列对话模型XVERSE-13B-256K ，该版本模型最大支持 256K 的上下文窗口长度，约 25w 字的输入内容，可以协助进行文献总结、报告分析等任务。
- [2023/12/14] 发布 XVERSE-65B-Chat 模型，该模型为基于 XVERSE-65B 底座模型的对齐版本。
- [2023/12/08] 发布 XVERSE-65B-2 底座模型，该模型在前一版本的基础上进行了 Continual Pre-Training，训练总 token 量达到 3.2 万亿；模型各方面的能力均得到提升，尤其是数学和代码能力，在 GSM8K 上提升 20%，HumanEval 上提升 41%。
- [2023/11/06] 发布 65B 尺寸的 XVERSE-65B 底座模型。
- [2023/11/06] 发布新版本的 XVERSE-13B-2 底座模型和 XVERSE-13B-2-Chat 对话模型，相较于原始版本，新版本的模型训练更加充分（从 1.4T 增加到 3.2T），各方面的能力均得到大幅提升，同时新增工具调用能力。
- [2023/09/26] 发布 7B 尺寸的 XVERSE-7B 底座模型和 XVERSE-7B-Chat 对话模型，支持在单张消费级显卡部署运行，并保持高性能、全开源、免费可商用。
- [2023/08/22] 发布经过指令精调的 XVERSE-13B-Chat 对话模型。
- [2023/08/07] 发布 13B 尺寸的 XVERSE-13B 底座模型。


# ToDo
Chat-GGUF 乱码， 估计是chat_template的问题

```
$ python -m applications.chat_cli run xverse/XVERSE-7B-Chat-GGUF+*q4_0.gguf

[对话第1轮]
(用户输入:)
你好
(xverse/XVERSE-7B-Chat-GGUF+*q4_0.gguf:)

，我是 <u>[NICK][@-NICK]?[@-SERVER]|[@-USER]> <nick,alias> [(options)] <text>
2. 命令格式： /msg <s>[INST]你好 [/INST]，我是 <u>[NICK][@-NICK]?[@-SERVER]|[@-USER]> <nick,alias> [(options)] <text>
3. 命令示例：/msg <s>server你好 [/INST]，我是 <u>username> nick [(options)] <text>
```


## License Agreement

使用本仓库的源码需要遵循 [Apache-2.0](https://github.com/xverse-ai/XVERSE-65B/blob/main/LICENSE) 开源协议，使用 XVERSE-65B 的模型权重则需要遵循[模型许可协议](https://github.com/xverse-ai/XVERSE-65B/blob/main/MODEL_LICENSE.pdf)。

XVERSE-65B 模型权重对学术研究**完全开放**，并且支持**免费商用**。如需申请商业许可证，请填写【[申请表](https://chat.xverse.cn/home/business.html)】，如有其他问题或合作，请联系 <opensource@xverse.cn>。

The use of the source code in this repository must follow the [Apache-2.0](LICENSE) open-source license, while the use of the model weights of XVERSE-65B needs to adhere to the [Model License Agreement](MODEL_LICENSE.pdf).

The XVERSE-65B model weights are **fully open** to academic research and support **free commercial use**.  To apply for a commercial license, please fill in the [application form](https://chat.xverse.cn/home/business.html). For other questions or collaborations, please contact <opensource@xverse.cn>.

## Reference
[Homepage](https://chat.xverse.cn/home/index.html)

[GITHUB](https://github.com/xverse-ai/)

[Hugging Face](https://huggingface.co/xverse)

