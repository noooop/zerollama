## Index-1.9B
Index-1.9B系列是Index系列模型中的轻量版本，包含以下模型： 
- Index-1.9B base : 基座模型，具有 19亿 非词嵌入参数量，在2.8T 中英文为主的语料上预训练，多个评测基准上与同级别模型比处于领先. 
- Index-1.9B pure : 基座模型的对照组，与base具有相同的参数和训练策略，不同之处在于我们严格过滤了该版本语料中所有指令相关的数据，以此来验证指令对benchmark的影响 
- Index-1.9B chat : 基于index-1.9B base通过SFT和DPO对齐后的对话模型，我们发现由于我们预训练中引入了较多互联网社区语料，聊天的<b>趣味性</b>明显更强 
- Index-1.9B character : 在SFT和DPO的基础上引入了RAG来实现<b>fewshots角色扮演</b>定制

# News
* 2024.06.13: 发布Index-1.9B系列


### License
使用本仓库的源码需要遵循 [Apache-2.0](LICENSE) 开源协议，使用 Index-1.9B 的模型权重则需要遵循[模型许可协议](INDEX_MODEL_LICENSE)。

Index-1.9B 模型权重对学术研究**完全开放**，并且支持**免费商用**。


### Reference
[GITHUB](https://github.com/bilibili/Index-1.9B)

[Hugging Face](https://huggingface.co/IndexTeam)

