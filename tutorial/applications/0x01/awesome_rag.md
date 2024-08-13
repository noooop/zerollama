
# Adaptive Retrieval / Query Classification
Not all queries require retrieval-augmented due to the inherent capabilities of LLMs.
Queries requiring retrieval proceed through the RAG modules; others are handled directly by LLMs.

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

- Mon, 1 Jul 2024 [Searching for Best Practices in Retrieval-Augmented Generation](https://arxiv.org/abs/2407.01219)
  - 将 query 二分类，需不需检索
  - BERT-base-multilingual Acc 0.95 Prec 0.96 Rec 0.94 F1 0.95

# Chunking / Chucking Granularity
- Thu, 22 Aug 2019 [Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering](https://arxiv.org/abs/1908.08167)
  - 实际测试，Natural Questions这类不是Multi-hop的数据集就是100 words的上下文就够了，对所有生成模型上下文越长，效果越差
  - 就是很迷，长上下文能不能支棱起来
- Mon, 5 Nov 2023 [Small-to-Big Retrieval](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4)
- Mon, 11 Dec 2023 [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://arxiv.org/abs/2312.06648)
- Wed, 31 Jan 2024 [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- Tue, 12 Mar 2024 [5 Levels Of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
- Mon, 20 May 2024 [Question-Based Retrieval using Atomic Units for Enterprise RAG](https://arxiv.org/abs/2405.12363)
- Fri, 21 Jun 2024 [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs](https://arxiv.org/abs/2406.15319)
- Tue, 25 Jun 2024 [LumberChunker: Long-Form Narrative Document Segmentation](https://arxiv.org/abs/2406.17526)
- Mon, 1 Jul 2024 [Searching for Best Practices in Retrieval-Augmented Generation](https://arxiv.org/abs/2407.01219)
  - Larger chunks can capture more context, but they also generate more noise, requiring longer processing time and higher costs. 
  - While smaller chunks may not fully convey the necessary context, they do have less noise
- Fri, 26 Jul 2024 [Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks](https://arxiv.org/abs/2407.21059)
  - 除了论文里提出，有名有姓的方法，Sliding Window、Metadata Attachment、Hierarchical Index也是简单有效的方法


# RAG for pre-train
- Fri, 22 May 2020 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Wed, 8 Dec 2021 [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)
- Tue, 31 Oct 2023 [GAR-meets-RAG Paradigm for Zero-Shot Information Retrieval](https://arxiv.org/abs/2310.20158)
- Fri, 8 Mar 2024 [PipeRAG: Fast Retrieval-Augmented Generation via Algorithm-System Co-design](https://arxiv.org/abs/2403.05676)

# Documentation Parsing
- Wed, 8 May 2024 [A Method for Parsing and Vectorization of Semi-structured Data used in Retrieval Augmented Generation](https://arxiv.org/abs/2405.03989)
  - unstructured -> .docx -> chunk by title -> RAG

# Fine-Tuning for RAG
随着大模型能力不断提升， zeroshot、in-Context Learning 和 follow instructions 的能力不断增强，专门为 RAG 任务 Fine-Tuning 到底有多大的收益

- Sun, 25 Jun 2023 [SAIL: Search-Augmented Instruction Learning](https://arxiv.org/abs/2305.15225)
  - We show that instruction-tuned LLMs can be heavily misled by distracting grounding information and noisy search results.
  - We constructed a search-augmented instruction training corpus.
  - We fine-tune a 7B-parameter language model (SAIL-7B) with the constructed training set, which outperforms strong baseline models including GPT-3.5-Turbo and Vicuna-13B on several NLP tasks.
