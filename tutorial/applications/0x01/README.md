# 检索增强生成 Retrieval-augmented generation (RAG)

## 1. 定义
大语言模型（LLMs）能力越来越强的同时，实际使用仍然有一些严重问题，尤其是在处理跟训练数据集显著不同的专业领域、严肃知识领域和需要最新消息时更容易产生“幻觉”，也就是胡说八道。

检索增强生成（RAG）是一项利用外部知识源提升大型语言模型（LLMs）的文本生成能力的技术。 通过引用外部知识，RAG 方法可以让大语言模型生成更准确、更可信、更有用的输出。

虽然可以通过在领域数据、实时数据上进行微调，但显然检索增强生成本倍更低，灵活性更高，更能快速响应实时变化的最新信息。同时也更容易集成到现有数据管理系统，进行有效的数据治理。

检索增强生成（RAG）以其成本低、灵活度高、多功能、可扩展、准确、可控、可解释等特性，得到越来越广泛的研究和使用。

## 2. RAG 系统总览
RAG 主要由离线和在线两个阶段组成

### 2.1. 离线 数据处理、分快、索引
1. ETL 清理原始数据，转换成大语音模型可以读取的格式，也就是纯文本格式
2. Chunking 为了适应语言模型的上下文长度，加快响应时间，文本被分为更小且更易于使用的块
3. Indexing 分快数据构建索引加快检索，可以针对需求微调、捕捉语义相似度的 Embedding 模型，配合索引系统向量数据库（vector database）成为新时代的标配。传统的检索特征比如 BM25 实际上仍然很能打，谁能拒绝全都要呢
  
### 2.2. 在线 检索、生成
1. Retrieval 用户查询首先从外部知识源检索出相关信息
2. Generation 用户查询和检索到的相关信息一起输入到大语言模型，希望大语言模型通过引用外部知识，生成更准确、更可信、更有用输出

### 2.3. 高级 RAG
随着大语言模型的发展和研究的深入，RAG 模块越来越多，功能越来越强。之后会介绍如何一步步升级 RAG 系统。

## 3. RAG 场景

问答系统不是唯一的 RAG 应用场景， 参考 [CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models](https://arxiv.org/abs/2401.17043) 
类比到数据库的 Create, Read, Update, and Delete (CRUD) 操作，总结了 RAG 的四种应用场景：

> - **Create** (generation) 在“增”场景下，系统通过参考知识库中的信息，丰富输入文本，生成创意输出，如诗歌、故事或代码。
> - **Read** (question-answering) 在“查”场景下，系统检索外部知识，推理并回答用户的问题。
> - **Update** (rectifying) 在“改”场景下，系统使用检索到的内容纠正输入文本中的错误，纠正拼写、语法或事实错误。 
> - **Delete** (summarizing) 在“删”场景下，系统简化检索到的信息，删除无关的不重要内容，呈现给用户更精炼的摘要。

举个例子

文生图需要提供提示词prompt描述想要生成的内容，随着经验的积累，无论是自己实验的发现，还是从网络搜刮而来，整合成一个 awesome prompts 集合。
现在，想做一个AI自动化流程，将新需求结合从awesome prompts 集合查询出来相似的数据，生成一些参考的prompt。这就是个 检索增强生成“增”场景。

## RAG 历史
检索增强生成（RAG）最初提出是2020年，当时还是做为一种预训练阶段增强技术。

2022 年 12 月 ChatGPT 的发布标志着 RAG 的一个重大转折。ChatGPT 的 in-Context Learning 和 follow instructions 的能力得到广泛的接受认可，
RAG 研究转向为在推理阶段通过外部知识，增强 llm 解决更复杂更知识密集型的任务。这也是现在普遍理解的 RAG 的形态。

不同模块用不同的目标分别训练，用不同的评价体系评估肯定是次优的选择。
随着研究的深入，开源大语言模型种类数量越来越多，能力也越来越强。
以及轻量化微调（Parameter-Efficient Fine-Tuning）技术使得微调成本越来越低，RAG 开始结合微调技术（fine-tuning），
毕竟谁也无法拒绝端到端训练一个系统，带来的解决方案的简洁优雅和性能的充分提升。




# Devils in the details

## 文档解析

### pdf




## RAG 流程

参考 [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)，可以将 RAG 流程分成四种类型

>- **Sequential** 顺序执行, Query-(pre-retrieval)-retriever-(post-retrieval)-generator
>- **Conditional** 条件执行，针对不同类型的输入查询使用不同的路径
>- **Branching** 并发执行，并发执行多个路径，合并每个路径的结果
>- **Loop**: 迭代执行，retriever-generator 到满意结果


## 4. The Good, the Bad and the Ugly

### 4.1. The Good 已经有非常的研究和开源模型代码

### 4.2. The Bad 还有很多问题没有解决，需要进一步研究

### 4.3. The Ugly 有很多脏活累活，无法自动化的流程

### Adaptive Retrieval Augmentation
[Self-Knowledge Guided Retrieval Augmentation for Large Language Models (EMNLP Findings 2023)](https://arxiv.org/abs/2310.05002)
[GITHUB](https://github.com/THUNLP-MT/SKR)
- 提出一种 Adaptive Retrieval Augmentation 方法，将 query 二分类，需不需检索
- 提出四种方法构造分类器
  - Direct Prompting，直接问llm
  - in-Context Learning，给llm几个例子
  - Training a Classifier，比如 BERT
  - Nearest Neighbor Search


[When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories](https://arxiv.org/abs/2212.10511)

### Query Rewriter
[Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2305.14283)
[GITHUB](https://github.com/xbmxb/RAG-query-rewriting)
- We present Rewrite-Retrieve-Read, a pipeline that improves the retrieval-augmented LLM from the perspective of query rewriting.
- Baselines:
- Direct: The standard in-context learning without any augmentations.
- Retrieve-then-read: The standard retrieval-augmented method. Retrieved documents are concatenated with the question.
- LLM as a frozen rewriter: As is introduced in §3.1, we prompt a frozen LLM to reason and generate queries by few-shot in-context learning. Our prompt follows the formulation of instruction, demonstrations, input
- Trainable rewriter: Applying the fine-tuned rewriter, the output queries are used by the retriever and the reader. 
  - The trainable model is initialized with the pre-trained T5-large (770M) 
  - Rewriter Warm-up. Inspired by recent distillation methods. we prompt the LLM to rewrite the original questions x in the training set and collect the generated queries x˜ as pseudo labels. The collected samples are then filtered
  - Reinforcement Learning. To further fine-tune the rewriter to cater to the LLM reader, we adopt a policy gradient reinforcement learning framework.
  - Policy Optimization We adopt Proximal Policy Optimization (PPO) (Schulman et al., 2017), following (Ramamurthy et al., 2022).


## Reference
[CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models](https://arxiv.org/abs/2401.17043) 
[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)