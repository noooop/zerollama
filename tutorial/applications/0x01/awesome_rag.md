
# RAG for pre-train
- Fri, 22 May 2020 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Wed, 8 Dec 2021 [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)
- Tue, 31 Oct 2023 [GAR-meets-RAG Paradigm for Zero-Shot Information Retrieval](https://arxiv.org/abs/2310.20158)
- Fri, 8 Mar 2024 [PipeRAG: Fast Retrieval-Augmented Generation via Algorithm-System Co-design](https://arxiv.org/abs/2403.05676)

# Documentation Parsing
- Wed, 8 May 2024 [A Method for Parsing and Vectorization of Semi-structured Data used in Retrieval Augmented Generation](https://arxiv.org/abs/2405.03989)
  - unstructured -> .docx -> chunk by title -> RAG

# Retrieval

- Fri, 23 Feb 2024 [Self-Retrieval: Building an Information Retrieval System with One Large Language Model](https://arxiv.org/abs/2403.00801)
  - LLM can memorize (passage -> title)
  - 至少Hit@1、 Hit@5、 MRR@5 指标比 dense retrieval 模型 GTR BGE OpenAI 效果好?? 所以 dense retrieval 必须配合 reranker ??


# Adaptive Retrieval
- Sun, 8 Oct 2023 [Self-Knowledge Guided Retrieval Augmentation for Large Language Models](https://arxiv.org/abs/2310.05002)
  - 将 query 二分类，需不需检索
  - 提出四种方法构造分类器
    - Direct Prompting，直接问llm
    - in-Context Learning，给llm几个例子
    - Training a Classifier，比如 BERT
    - Nearest Neighbor Search

- Tue, 30 Apr 2024 [When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively](https://arxiv.org/abs/2404.19705)
  - 分类学 Adaptive Retrieval、Never-Retrieve (NR-LLM)、Always-Retrieve (AR-LLM)
  - LLMs are trained to generate a special token, ⟨RET⟩, when they do not know the answer to a question

# Fine-Tuning for RAG
随着大模型能力不断提升， zeroshot、in-Context Learning 和 follow instructions 的能力不断增强，专门为 RAG 任务 Fine-Tuning 到底有多大的收益

- Sun, 25 Jun 2023 [SAIL: Search-Augmented Instruction Learning](https://arxiv.org/abs/2305.15225)
  - We show that instruction-tuned LLMs can be heavily misled by distracting grounding information and noisy search results.
  - We constructed a search-augmented instruction training corpus.
  - We fine-tune a 7B-parameter language model (SAIL-7B) with the constructed training set, which outperforms strong baseline models including GPT-3.5-Turbo and Vicuna-13B on several NLP tasks.
