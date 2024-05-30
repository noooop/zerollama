# Yuan (源)
浪潮信息最新一代开源基础大模型


# Yuan2.0-M32: Mixture of Experts with Attention Router

##  TL;DR

浪潮信息 **“源2.0 M32”大模型（简称，Yuan2.0-M32）** 采用稀疏混合专家架构（MoE），以Yuan2.0-2B模型作为基底模型，通过创新的门控网络（Attention Router）实现32个专家间（Experts*32）的协同工作与任务调度，在显著降低模型推理算力需求的情况下，带来了更强的模型精度表现与推理性能；源2.0-M32在多个业界主流的评测进行了代码生成、数学问题求解、科学问答与综合知识能力等方面的能力测评。结果显示，源2.0-M32在多项任务评测中，展示出了较为先进的能力表现，MATH（数学求解）、ARC-C（科学问答）测试精度超过LLaMA3-700亿模型。**Yuan2.0-M32大模型** 基本信息如下：

+ **模型参数量：** 40B <br>
+ **专家数量：** 32 <br>
+ **激活专家数：** 2 <br>
+ **激活参数量：** 3.7B <br>  
+ **训练数据量：** 2000B tokens <br>
+ **支持序列长度：** 16K <br>


同时，我们发布了Yuan2.0-M32模型的<a href="https://arxiv.org/abs/2405.17976" target="_blank">**技术报告**</a>，可以通过论文查看更详细的技术细节与测评结果。

## Todo
ValueError: Multiple files found in IEITYuan/Yuan2-M32-gguf-int4 matching *.gguf

## News
* [2024-05-28] 发布源2.0 M32 大模型

## License Agreement

The use of the source code in this repository requires compliance with the open source license agreement Apache 2.0. The Yuan2.0 model supports commercial use and does not require authorization. Please understand and comply with the 《Yuan2.0 Model License Agreement》. Do not use the open source model and code, as well as derivatives generated from open source projects, for any purposes that may cause harm to the country and society, or for any services that have not undergone security assessment and filing. Although we have taken measures to ensure the compliance and accuracy of the data during training, the model has a huge number of parameters and is affected by probability and randomness factors. We cannot guarantee the accuracy of the output content, and the model is easily misled by input instructions. This project does not assume any data security, public opinion risks, or any model misleading, abusing, spreading caused by open-source models and code Risks and responsibilities arising from improper utilization You will be solely responsible for the risks and consequences arising from the use, copying, distribution, and modification of the model in this open source project

使用源2.0代码及模型需遵循 [Apache 2.0](https://github.com/xxxxxxE) 开源协议和[《源2.0模型许可协议》](./LICENSE-Yuan)，源2.0模型支持商用，不需要申请授权，请您了解并遵循，勿将开源模型和代码及基于开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。

尽管模型在训练时我们已采取措施尽力确保数据的合规性和准确性，但模型参数量巨大且受概率随机性因素影响，我们无法保证输出内容的准确性，且模型易被输入指令所误导，本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。**您将对通过使用、复制、分发和修改模型等方式利用该开源项目所产生的风险与后果，独自承担全部责任。**


## Reference
[GITHUB](https://github.com/IEIT-Yuan/Yuan2.0-M32)

[Hugging Face](https://huggingface.co/IEITYuan)

[ModelScope](https://www.modelscope.cn/profile/YuanLLM)

[Tech Report](https://arxiv.org/abs/2405.17976)

