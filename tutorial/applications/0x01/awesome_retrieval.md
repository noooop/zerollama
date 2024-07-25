
# Overview
借 [ColBERT](https://arxiv.org/abs/2004.12832) 的图，广义的计算句子之间相似度的模型大概有四种架构：

<img src="https://github.com/noooop/noooop.github.io/blob/main/applications/rag/colbert.png?raw=true" width="400">

其中第一种被称为“双塔模型(dual-encoder)”，Query 和 Document 分别做 Embeddings，特征交互只发生在最后，一般是简单的算两个Embeddings的余弦距离。
Document 的 Embeddings 可以离线预计算，Embeddings存储量比较小，线上只需要算简单的余弦距离，还可以用ANN (Approximate Nearest Neighbor) 加速。
可以scale，从几万甚至百万千万文档召回。

后面几种 Query 和 Document 特征从浅层开始融合，效果肯定比第一种“双塔模型”好，但从浅层开始融合意味着需要Query和Document两两计算，计算量也很大，没办法scale。

retrieval rerank 两阶段检索，第一阶段先用双塔模型大量召回比如，top100,，第二阶段将召回候选集和Query两两计算，得到更精确的检索排序。


# Traditional Retrieval(Sparse lexical search algorithms)
虽然 dense retrieval 从2020年开始变成检索模型的主流，传统检索算法比如 BM25 对关键词、专业名词等召回效果比较好，仍然是 dense retrieval 有效的补充。
- [Which bm25 do you mean? a large-scale reproducibility study of scoring variants](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_4)
  - Chris Kamphuis, Arjen P De Vries, Leonid Boytsov, and Jimmy Lin. 2020. : 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, 
  - In Advances in Information Retrieval Proceedings, Part II 42, pages 28–34. Springer.
- Thu, 4 Jul 2024 [BM25S: Orders of magnitude faster lexical search via eager sparse scoring](https://arxiv.org/abs/2407.03618)
  - We introduce BM25S, an efficient Python-based implementation of BM25 that only depends on Numpy and Scipy.
  - It also achieves considerable speedups compared to highly optimized Java-based implementations, which are used by popular commercial products.
  - 单线程跟 lucene 有的一拼，速度真的很快

# History
诡异的数据集构建和测试方法怎么来的。
- Thu, 22 Aug 2019 [Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering](https://arxiv.org/abs/1908.08167)
  - previous work defines passages as articles, paragraphs, or sentences. However, the question of proper granularity of passages is still underexplored.
    - (RAG chucking 的粒度问题从 2019 年一直讨论到 2024 年，base模型的推理能力和上下文能力每增强一次都会重新讨论一次
  - we find that splitting articles into passages with the length of 100 words by sliding window improves performance by 4%.
    - We set the window size as 100 words, and the stride as 50 words(half the window size). 
    - 记住这个 100 words as passages
  - Passage ranker reranks all retrieved passages, and selects a list of high-quality passages for the multi-passage BERT model.
    - First, the retriever returns top-100 passages for each question. Then, the passage ranker is employed to rerank these 100 passages. Finally, multi-passage BERT takes top30 reranked passages as input to pinpoint the final answer. 
    - (ranker 理念出现的时间也非常早
  - we use the 2016-12-21 English Wikipedia dump. Following DrQA 2017


# Retrieval(Embeddings) model
- Fri, 10 Apr 2020 [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
  - DPR 论文提出的整个体系，包括模型、训练、在线推理，都跟现在主流相近，数据集的处理方式最新论文还在follow，在几个数据集上的结果最新论文仍作为baseline比较。那就多写一点。
  - Transformer 2017年发布，BERT 2019年发布，开始刷nlp任务。真是勃勃生机万物竞发的时代。
  - 使用BERT预训练模型，Embeddings 维度 768，Query 和 Document使用不同模型，也就是要训练两个模型，inner product 作为相似函数， loss function 使用 negative log likelihood
  - DPR performs consistently better than BM25 on all datasets.
  - Experimental Setup:
    - English Wikipedia dump from Dec. 20, 2018 as the source documents for answering questions. Following (Lee et al., 2019), Following DrQA 2017 
    - We then split each article into multiple, disjoint text blocks of 100 words as passages, serving as our basic retrieval units. Following (Wang et al., 2019) 
      - 记住这个 100 words as passages
    - five QA datasets(Natural Questions 2019(NQ), TriviaQA 2017, WebQuestions 2013(WQ), CuratedTREC 2015(TREC), SQuAD v1.1 2016)
    - Selection of positive passages
      - TREC, WebQuestions and TriviaQA:  we use the highest-ranked passage from BM25 that contains the answer as the positive passage. If none of the top 100 retrieved passages has the answer, the question will be discarded.
      - SQuAD and Natural Questions: since the original passages have been split and processed differently than our pool of candidate passages, we match and replace each gold passage with the corresponding passage in the candidate pool.
  - Ablation Study on Model Training
    - Sample efficiency 
      - a dense passage retriever trained using only 1,000 examples already outperforms BM25.
      - Adding more training examples (from 1k to 59k) further improves the retrieval accuracy consistently.
    - In-batch negative training
      - re-using gold passages from the same batch as negatives can make the computation efficient while achieving great performance.
      - a batch size of 128 and one additional BM25 negative passage per question
      - in-batch negative training improves the results substantially. As a result, accuracy consistently improves as the batch size grows. 
      - “hard” negative passages that have high BM25 scores given the question, but do not contain the answer string (the bottom block)
      - We find that adding a single BM25 negative passage improves the result substantially while adding two does not help further.
    - Impact of gold passages 
    - Similarity and loss
      - L2 performs comparable to dot product, and both of them are superior to cosine.
      - Our experiments show that using triplet loss does not affect the results much.
    - Cross-dataset generalization
    - Qualitative Analysis
      - Term-matching methods like BM25 are sensitive to highly selective keywords and phrases
      - while DPR captures lexical variations or semantic relationships better.
    - Run-time Efficiency
      - BM25+Lucene vs DPR+FAISS
  - End-to-end QA System
    - The probabilities of a token being the starting/ending positions of an answer span and a passage being selected. (这种预测答案位置的 reading comprehension (RC) 任务已经退出历史舞台)
    - measured by exact match with the reference answer after minor normalization as in (Chen et al., 2017; Lee et al., 2019)  (EM的评判标准一直延续下来）
    - higher retriever accuracy typically leads to better final QA results
    - Recent work (Izacard and Grave, 2020; Lewis et al., 2020b) have also shown that DPR can be combined with generation models such as BART (Lewis et al., 2020a) and T5 (Raffel et al., 2019), achieving good performance on open-domain QA and other knowledge-intensive tasks.
    - Retrieval + generation 这已经很RAG了

  - <img src="https://github.com/noooop/noooop.github.io/blob/main/applications/rag/dpr1.png?raw=true" width="400">
  - <img src="https://github.com/noooop/noooop.github.io/blob/main/applications/rag/dpr2.png?raw=true" width="400">


# Chucking Granularity
- Thu, 22 Aug 2019 [Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering](https://arxiv.org/abs/1908.08167)
- Mon, 11 Dec 2023 [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://arxiv.org/abs/2312.06648)
- Wed, 31 Jan 2024 [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- Mon, 20 May 2024 [Question-Based Retrieval using Atomic Units for Enterprise RAG](https://arxiv.org/abs/2405.12363)
- Fri, 21 Jun 2024 [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs](https://arxiv.org/abs/2406.15319)
- Tue, 25 Jun 2024 [LumberChunker: Long-Form Narrative Document Segmentation](https://arxiv.org/abs/2406.17526)
  
# Rerank model
- Tue, 27 Aug 2019 [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
  - [sentence-transformers](https://github.com/UKPLab/sentence-transformers/)
  - [Document](https://www.sbert.net/)
- Mon, 27 Apr 2020 [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)

# Other
- Fri, 23 Feb 2024 [Self-Retrieval: Building an Information Retrieval System with One Large Language Model](https://arxiv.org/abs/2403.00801)
  - LLM can memorize (passage -> title)
  - 至少Hit@1、 Hit@5、 MRR@5 指标比 dense retrieval 模型 GTR BGE OpenAI 效果好?? 所以 dense retrieval 必须配合 reranker ??
