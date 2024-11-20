
# Routing
In response to diverse queries, the RAG system routes to specific pipelines tailored for different scenario, a feature essential for a versatile RAG architecture designed to handle a wide array of situations.


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

# Chunking / Chucking Granularity / Chunking Techniques
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
- Sat, 7 Sep 2024 [Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models](https://arxiv.org/abs/2409.04701)
  - 先过transformer layers，在sequence_output上做Chunking mean pooling，有意思
- Wed, 16 Oct 2024 [Is Semantic Chunking Worth the Computational Cost?](https://arxiv.org/abs/2410.13070)
  - We demonstrate that while semantic chunking shows some benefits in certain scenarios,
  these are inconsistent and often insufficient to
  justify the computational cost.

# Retrieval / Embedding Model
[跳转](./awesome_retrieval.md#retrievalembeddings-model)

# Vector Databases
- BruteForce 对于几个pdf的数据量，硬算速度也很快
- Milvus
- Chroma
- Faiss
- hnswlib
- Tue, 10 Sep 2024 [Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes?](https://arxiv.org/abs/2409.06464)
  - based on Lucene 9.9.1 (released Dec. 2023)
  - https://www.elastic.co/cn/blog/whats-new-elasticsearch-platform-8-13-0 至少Elasticsearch 8.0 就支持 HNSW 和 flat
  - 问题就来了，Table 2 看不出随着数据量增加，HNSW 和 flat 召回如何变化，期待更详细的测试
- https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/ongoing_leaderboard/leaderboard.md
  - Pinecone 和 Zilliz 真的好猛

# Query Rewriting\Query Disambiguation\Query Decomposition
- Thu, 30 Apr 2020 [Question Rewriting for Conversational Question Answering](https://arxiv.org/abs/2004.14652)
- Tue, 19 Jan 2021 [A Comparison of Question Rewriting Methods for Conversational Passage Retrieval](https://arxiv.org/abs/2101.07382)
- Sat, 22 Jan 2022 [Question rewriting? Assessing its importance for conversational question answering](https://arxiv.org/abs/2201.09146)
- Tue, 23 May 2023 [Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2305.14283)
- Tue, 7 Nov 2023 [Large Language Model based Long-tail Query Rewriting in Taobao Search](https://arxiv.org/abs/2311.03758)
- Tue, 19 Dec 2023 [Rewriting Conversational Utterances with Instructed Large Language Models](https://ieeexplore.ieee.org/document/10350178)
- Thu, 18 Jan 2024 [ChatQA: Surpassing GPT-4 on Conversational QA and RAG](https://arxiv.org/abs/2401.10225)
  - Query Rewriting 并没有增加新的信息，最新的 Retrieval(Embedding) Model 大小来到了7B 甚至更大，基础模型也使用LLM as Retrieval。
  - Conversational Query Rewriting Most of the previous solutions are query rewriting methods.
    - The latest turn of question is rewritten to be a standalone query without additional information from previous dialogue history (Vakulenko et al., 2021a; Ye et al., 2023; Mo et al., 2023), 
    - so it can be directly used by retrieval model to retrieve relevant context (Vakulenko et al., 2021b; Mele et al., 2021; Raposo et al., 2022; Mo et al., 2023).
    - 使用大 retrieval model 的效果与使用 GPT-3.5-Turbo Query Rewriting 相当
  - fine-tuning a good single-turn retriever on high-quality conversational query context pairs performs on par with leveraging the state-of-the-art rewriter.
  - However, rewriting method requires extra computational time for autoregressive generation process and probably also API cost for using powerful models like GPT-3.5-Turbo. 
- Mon, 1 Jul 2024 [Searching for Best Practices in Retrieval-Augmented Generation](https://arxiv.org/abs/2407.01219)
  - However, query rewriting and query decomposition did not enhance retrieval performance as effectively
- Sat, 20 Jul 2024 [Golden-Retriever: High-Fidelity Agentic Retrieval Augmented Generation for Industrial Knowledge Base](https://arxiv.org/abs/2408.00798)
- Mon, 9 Sep 2024 [MemoRAG: Moving towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery](https://arxiv.org/abs/2409.05591)
  - RQ-RAG 效果比较差

# Reranking 
[跳转](./awesome_retrieval.md#rerank-model)

- Fri, 26 Jul [Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks](https://arxiv.org/abs/2407.21059)
  - 使用通用大模型做 Reranking 也不是不行，就是有的慢
  - Another straightforward and effective approach involves having the LLM evaluate the retrieved content before generating the final answer

# Summarization
Retrieval results may contain redundant or unnecessary information. Additionally, long prompts can slow down the inference process.

Summarization tasks can be extractive or abstractive.
- Fri, 6 Oct 2023 [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408)
- Tue, 10 Oct 2023 [LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression](https://arxiv.org/abs/2310.06839)

# Document Repacking
The performance of subsequent processes, such as LLM response generation, may be affected by the order documents are provided.
- Thu, 6 Jul 2023 [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
  - 将重要的文档排到两边，大模型会偷懒不看中间的文档
  - 太诡异了，如果(2024年)新训练的，标榜长上下文的模型还需要将检索文档重新排序，那就真的太诡异了。


# Generate

# Verification
Given the input question, the retrieved knowledge, and the generated answer, a small language model is trained to determine whether the generated answer correctly reflects the retrieved knowledge.
- Wed, 24 May 2023 [SAIL: Search-Augmented Instruction Learning](https://arxiv.org/abs/2305.15225)
  - We show that instruction-tuned LLMs can be heavily misled by distracting grounding information and noisy search results.
  - 微调模型
- Tue, 17 Oct 2023 [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
  - 微调模型
- Thu, 19 Oct 2023 [Knowledge-Augmented Language Model Verification](https://arxiv.org/abs/2310.12836)
- Mon, 29 Jan 2024 [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
- Thu, 11 Jul 2024 [Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting](https://arxiv.org/abs/2407.08223)
  - Diversity and reduced redundancy in retrieval improves draft quality significantly. 检索的noise对生成的影响还是挺大的
  - ∗We use the RAG drafter’s generation probability ρDraft as the confidence score for selecting drafts when we use it alone
    - we use Mistral7B or Mixtral8x7B as the RAG verifier, and denote them as MVerifier-7B or MVerifier-8x7B
    - MDrafter-7B 是有什么魔法吗？
    - Table 1 TriviaQA 
      - Mistral7B 54.15 Mistral-Instruct7B 67.11 -> MDrafter-7B∗ 71.11 ？
    - Table 1 PubHealth 
      - Mistral7B  34.85 Mistral-Instruct7B 42.15 -> MDrafter-7B∗ 75.58 ？


# Advanced RAG\Agentic RAG
利用llm的Planning&reasoning, 做RAG复杂流程调度
- Tue, 17 Oct 2023 [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)


# Best Practice
- Thu, 18 Jan 2024 [ChatQA: Surpassing GPT-4 on Conversational QA and RAG](https://arxiv.org/abs/2401.10225)
- Mon, 1 Jul 2024 [Searching for Best Practices in Retrieval-Augmented Generation](https://arxiv.org/abs/2407.01219)
- Fri, 19 Jul 2024 [ChatQA 2: Bridging the Gap to Proprietary LLMs in Long Context and RAG Capabilities](https://arxiv.org/abs/2407.14482)

# Survey 
- Fri, 26 Jul 2024 [Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks](https://arxiv.org/abs/2407.21059)
- Mon, 23 Sep 2024 [Retrieval Augmented Generation (RAG) and Beyond: A Comprehensive Survey on How to Make your LLMs use External Data More Wisely](https://arxiv.org/abs/2409.14924)
  - Level-1 Explicit Facts
  - Level-2 Implicit Facts
  - Level-3 Interpretable Rationales
  - Level-4 Hidden Rationales

# Benchmark
- Mon, 4 Sep 2023 [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431)
  - Noise Robustness
  - Negative Rejection
  - Information Integration
  - Counterfactual Robustness
- Fri, 7 Jun 2024 [CRAG -- Comprehensive RAG Benchmark](https://arxiv.org/abs/2406.04744)
  - [kdd-cup-2024](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024)
  - Task
    - Task 1: Retrieval Summarization.
    - Task 2: KG and Web Retrieval Augmentation.
    - Task 3: End-to-end RAG.
  - Question type
    - Simple \ Simple w. condition \ Set \ Comparison \ Aggregation \ Multi-hop \ Post-processing heavy \ False Premise
  - Metrics
    -  We then use a scoring method Scoreh with score 1, 0.5, 0, and -1 for each perfect, acceptable, missing, and incorrect answer, respectively

# Evaluate
- Mon, 29 May 2023 [A Critical Evaluation of Evaluations for Long-form Question Answering](https://arxiv.org/abs/2305.18201)
  - We encourage future work to move away from a single “overall score” of the answer and adopt a multi-faceted evaluation, targeting aspects such as factuality and completeness.
- Mon, 20 May 2024 [xFinder: Robust and Pinpoint Answer Extraction for Large Language Models](https://arxiv.org/abs/2405.11874)
  - RegEx accuracy in the best evaluation framework is 74.38%, we propose xFinder, a model specifically designed for key answer extraction.


# Fine-Tuning for RAG
随着大模型能力不断提升， zeroshot、in-Context Learning 和 follow instructions 的能力不断增强，专门为 RAG 任务 Fine-Tuning 到底有多大的收益
- Wed, 24 May 2023 [SAIL: Search-Augmented Instruction Learning](https://arxiv.org/abs/2305.15225)
  - We show that instruction-tuned LLMs can be heavily misled by distracting grounding information and noisy search results.
  - We constructed a search-augmented instruction training corpus.
  - We fine-tune a 7B-parameter language model (SAIL-7B) with the constructed training set, which outperforms strong baseline models including GPT-3.5-Turbo and Vicuna-13B on several NLP tasks.
- Thu, 17 Oct 2024 [Block-Attention for Efficient RAG](https://arxiv.org/abs/2409.15355)
  - Block-Attention enables us to reuse the KV states of passages that have been seen before, thereby significantly reducing the latency and the computation overhead during inference

# Add control token
- Tue, 17 Oct 2023 [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
  - Agentic RAG?
    - Retrieve Decides when to retrieve with R.
    - ISREL d provides useful information to solve x.
    - ISSUP All of the verification-worthy statement in y is supported by d.
    - ISUSE y is a useful response to x.
- Mon, 6 May 2024 [When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively](https://arxiv.org/abs/2404.19705)
  - LLMs are trained to generate a special token, ⟨RET⟩, when they do not know the answer to a question. 

# Knowledge Graph RAG
- Wed, 24 Apr 2024 [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)
- Fri, 7 Jun 2024 [CRAG -- Comprehensive RAG Benchmark](https://arxiv.org/abs/2406.04744)
- Fri, 9 Aug 2024 [A Hybrid RAG System with Comprehensive Enhancement on Complex Reasoning](https://arxiv.org/abs/2408.05141)
  - CRAG + 1
- Tue, 8 Oct 2024 [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779)

# Long-Context Evaluate
- Wed, 21 Feb 2024 [∞Bench: Extending Long Context Evaluation Beyond 100K Tokens](https://arxiv.org/abs/2402.13718)
  - LRA (Tay et al., 2020) ∼10K 
  - LongBench (Bai et al., 2023) ∼10K QA, summarization, synthetic retrieval, and code
  - L-Eval (An et al., 2023) 4K - 60K QA, summarization, math, retrieval, and multiple-choice (MC)
  - LooGLE (Li et al., 2023) ∼20K summary and QA
  - ∞BENCH (ours) ∼200K 
    - Retrieve.PassKey 
    - Retrieve.Number
    - Retrieve.KV
    - En.Sum
    - En.QA
    - En.MC
    - En.Dia
    - Zh.QA
    - Code.Debug
    - Code.Run
    - Math.Calc
    - Math.Find
- Tue, 9 Apr 2024 [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654)
  - https://github.com/NVIDIA/RULER 
  - Retrieval: Needle-in-a-haystack (NIAH)
    - Single NIAH (S-NIAH)
    - Multi-keys NIAH (MK-NIAH)
    - Multi-values NIAH (MV-NIAH)
    - Multi-queries NIAH (MQ-NIAH)
  - Multi-hop Tracing: Variable Tracking (VT)
  - Aggregation: Common Words (CWE) and Frequent Words Extraction (FWE)
  - Question Answering (QA)
  - GLM4 (9B) 效果不错
- Sat, 29 Jun 2024 [Is It Really Long Context if All You Need Is Retrieval? Towards Genuinely Difficult Long Context NLP](https://arxiv.org/abs/2407.00402)
  - 


# Long-Context
- Tue, 27 Aug 2024 [Writing in the Margins: Better Inference Pattern for Long Context Retrieval](https://arxiv.org/abs/2408.14906)
  - Needle Retrieval and Single-Hop Question Answering
    - Unsurprisingly, RAG emerges as the most optimal pattern for 
six out of seven evaluated models when extending the context length to 64k tokens in SQuAD. Indeed, for single-hop
reasoning tasks, if the filtering process is successful (here we
approximate the retriever by an LLM classifier), the challenge
is reduced to a trivial task of retrieving a needle from a context window of 4096 tokens. However, this assumption in the
RAG setup is overly optimistic because the LLMs used in our
experiment are at least 7B in model parameters, and such large
models are not typically used as retrievers. 
    - 反过来说，如果成本可控，用7B以上的模型做retrievers几乎完美
- Mon, 9 Sep 2024 [MemoRAG: Moving towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery](https://arxiv.org/abs/2409.05591)
  - LongBench  	InfBench 	UltraDomain
  - Using Long LLMs as Memory Model
  - it employs a light but longrange LLM to form the global memory of database, generates draft answers
  - it leverages an expensive but expressive LLM, which generates the ultimate answer based on the retrieved information.
- Fri, 20 Sep 2024 [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
  - A note on simply using a longer prompt
- Sun, 6 Oct 2024 [Inference Scaling for Long-Context Retrieval Augmented Generation](https://arxiv.org/abs/2410.04343)
  - IterDRAG 只用从 32k -> 128k 才能超越 RAG 和 DRAG
  - Agentic RAG 需要非常大的 Inference Scaling 才能体现出优势
- Tue, 5 Nov 2024 [Long Context RAG Performance of Large Language Models](https://arxiv.org/abs/2411.03538)
  - 给模型更多512token的片段，而不是更长的连续上下文？？？？

# approximate Long-Context
使用短上下文近似长上下文效果， 相当于一种特别的 Agentic RAG
- Sun, 18 Feb 2024 [LongAgent: Scaling Language Models to 128k Context through Multi-Agent Collaboration](https://arxiv.org/abs/2402.11550)
  - Needle-in-a-Haystack PLUS (Single-Doc QA + Multi-Doc QA) 
- Fri, 24 May 2024 [Are Long-LLMs A Necessity For Long-Context Tasks?](https://arxiv.org/abs/2405.15318)
  - Single-Doc QA Multi-Doc QA + Summarization + Synthetic Task: Passage Count Self-Constructed Dataset + Code Completion
  - Hand-Designed Agent
    - Task Understanding
    - Retrieve
    - Move
    - Append
    - Merge
    - Answer
    - Aggregation
- Tue, 4 Jun 2024 [Chain of Agents: Large Language Models Collaborating on Long-Context Tasks](https://arxiv.org/abs/2406.02818)
  - QA Summarization Code Completion
- Sat, 12 Oct 2024 [LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models](https://arxiv.org/abs/2410.09342)
  - ∞BENCH
  - Map Stage + Collapse Stage + Reduce Stage
  - However, in complex cases where historical
and current information conflict, LC-Boost may
struggle to fully resolve the issue relying solely on
the accumulated summary and the current text.

# Practice
- Wed, 8 May 2024 [A Method for Parsing and Vectorization of Semi-structured Data used in Retrieval Augmented Generation](https://arxiv.org/abs/2405.03989)
  - unstructured -> .docx -> chunk by title -> RAG
- Mon, 24 Jun 2024 [Context-augmented Retrieval: A Novel Framework for Fast Information Retrieval based Response Generation using Large Language Model](https://arxiv.org/abs/2406.16383)
  - Query-Index Classification Model: The query is classified into the appropriate domain/category using the classification model
    - DistilBERT Training Accuracy 1.00 Testing Accuracy 0.42 Recall 0.42 Precision 0.59 ???
  - Entity Extraction, BM25 retriever, Hybrid Retriever
  - Index Generation: OpenAI’s text-embedding-ada-002 embeddings with a chunk size limit of 512 tokens
- Fri, 20 Sep 2024 [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
  - A note on simply using a longer prompt
    - Sometimes the simplest solution is the best. If your knowledge base is smaller than 200,000 tokens (about 500 pages of material), you can just include the entire knowledge base in the prompt that you give the model, with no need for RAG or similar methods.
  - Using Prompt Caching
  - BM25 + semantic embeddings
  - We’ve written a prompt that instructs the model to provide concise, chunk-specific context that explains the chunk using the context of the overall document.
    - WHOLE_DOCUMENT + CHUNK_CONTENT -> llm -> contextual text
  - Using Prompt Caching to reduce the costs of Contextual Retrieval
  
# RAG for pre-train
- Fri, 22 May 2020 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Wed, 8 Dec 2021 [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)
- Tue, 31 Oct 2023 [GAR-meets-RAG Paradigm for Zero-Shot Information Retrieval](https://arxiv.org/abs/2310.20158)
- Fri, 8 Mar 2024 [PipeRAG: Fast Retrieval-Augmented Generation via Algorithm-System Co-design](https://arxiv.org/abs/2403.05676)


