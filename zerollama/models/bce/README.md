# BCEmbedding

Netease Youdao's open-source embedding and reranker models for RAG products.

##  TL;DR
`BCEmbedding`是由网易有道开发的中英双语和跨语种语义表征算法模型库，其中包含 `EmbeddingModel`和 `RerankerModel`两类基础模型。`EmbeddingModel`专门用于生成语义向量，在语义搜索和问答中起着关键作用，而 `RerankerModel`擅长优化语义搜索结果和语义相关顺序精排。

`BCEmbedding`作为有道的检索增强生成式应用（RAG）的基石，特别是在[QAnything](http://qanything.ai) [[github](https://github.com/netease-youdao/qanything)]中发挥着重要作用。QAnything作为一个网易有道开源项目，在有道许多产品中有很好的应用实践，比如[有道速读](https://read.youdao.com/#/home)和[有道翻译](https://fanyi.youdao.com/download-Mac?keyfrom=fanyiweb_navigation)。

`BCEmbedding`以其出色的双语和跨语种能力而著称，在语义检索中消除中英语言之间的差异，从而实现：

- **强大的双语和跨语种语义表征能力【<a href="#基于mteb的语义表征评测指标" target="_Self">基于MTEB的语义表征评测指标</a>】。**
- **基于LlamaIndex的RAG评测，表现SOTA【<a href="#基于llamaindex的rag评测指标" target="_Self">基于LlamaIndex的RAG评测指标</a>】。**

## 开源目的

给RAG社区一个可以直接拿来用，尽可能不需要用户finetune的中英双语和跨语种二阶段检索模型库，包含`EmbeddingModel`和`RerankerModel`。

- 只需一个模型：`EmbeddingModel`覆盖 **中英双语和中英跨语种** 检索任务，尤其是其跨语种能力。`RerankerModel`支持 **中英日韩** 四个语种及其跨语种。
- 只需一个模型： **覆盖常见业务落地领域**（针对众多常见rag场景已做优化），比如：教育、医疗、法律、金融、科研论文、客服(FAQ)、书籍、百科、通用QA等场景。用户不需要在上述特定领域finetune，直接可以用。
- 方便集成：`EmbeddingModel`和`RerankerModel`提供了LlamaIndex和LangChain **集成接口** ，用户可非常方便集成进现有产品中。
- 其他特性：
  - `RerankerModel`支持 **长passage（超过512 tokens，不超过32k tokens）rerank**；
  - `RerankerModel`可以给出有意义 **相关性分数** ，帮助 **过滤低质量召回**；
  - `EmbeddingModel` **不需要“精心设计”instruction** ，尽可能召回有用片段。

### requirements
BCEmbedding==0.1.5

### News
- ***2024-02-04***: **BCEmbedding技术博客** - 包含算法设计和实操细节，<a href="https://zhuanlan.zhihu.com/p/681370855">为RAG而生-BCEmbedding技术报告</a>。
- ***2024-01-16***: **LangChain和LlamaIndex集成** - 详见<a href="#embedding和reranker集成常用rag框架" target="_Self">演示样例</a>。
- ***2024-01-03***: **模型发布** - [bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1)和[bce-reranker-base_v1](https://huggingface.co/maidalun1020/bce-reranker-base_v1)已发布.
- ***2024-01-03***: **RAG评测数据** [[CrosslingualMultiDomainsDataset](https://huggingface.co/datasets/maidalun1020/CrosslingualMultiDomainsDataset)] - 基于[LlamaIndex](https://github.com/run-llama/llama_index)的RAG评测数据已发布。
- ***2024-01-03***: **跨语种语义表征评测数据** [[详情](./BCEmbedding/evaluation/c_mteb/Retrieval.py)] - 基于[MTEB](https://github.com/embeddings-benchmark/mteb)的跨语种评测数据已发布.

### 注意 
To download the weights from Hugging Face, You need to agree to share your contact information to access this model.

BCEmbedding 模型下载需要注册huggingface账户、填写申请、配置HF_TOKEN  

### License
BCEmbedding is licensed under Apache 2.0 License


### Reference
[GITHUB](https://github.com/netease-youdao/BCEmbedding/)

[Hugging Face](https://huggingface.co/maidalun1020/)
