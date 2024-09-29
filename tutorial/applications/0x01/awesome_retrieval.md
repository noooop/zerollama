
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
- 2009 [The probabilistic relevance framework: Bm25 and beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
  - Stephen Robertson, Hugo Zaragoza, et al. 2009.
  - Foundations and Trends in Information Retrieval, 3(4):333–389.
- 2020 [Which bm25 do you mean? a large-scale reproducibility study of scoring variants](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_4)
  - Chris Kamphuis, Arjen P De Vries, Leonid Boytsov, and Jimmy Lin. 2020. : 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, 
  - In Advances in Information Retrieval Proceedings, Part II 42, pages 28–34. Springer.
- Thu, 4 Jul 2024 [BM25S: Orders of magnitude faster lexical search via eager sparse scoring](https://arxiv.org/abs/2407.03618)
  - We introduce BM25S, an efficient Python-based implementation of BM25 that only depends on Numpy and Scipy.
  - It also achieves considerable speedups compared to highly optimized Java-based implementations, which are used by popular commercial products.
  - 单线程跟 lucene 有的一拼，速度真的很快

# Hybrid Retrievers
- Fri, 22 Mar 2024 [Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers](https://arxiv.org/abs/2404.07220)
- Mon, 1 Jul 2024 [Searching for Best Practices in Retrieval-Augmented Generation](https://arxiv.org/abs/2407.01219)
  - Taking efficiency into consideration, Hybrid Search combines sparse retrieval (BM25) and dense retrieval (Original embedding) and achieves notable performance with relatively low latency.

# History
诡异的数据集构建和测试方法怎么来的。
- Fri, 31 Mar 2017 [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)
  - Machine reading at scale (MRS).
    - Using Wikipedia articles as the knowledge source causes the task of question answering (QA) to combine the challenges of both large-scale open-domain QA and of machine comprehension of text.
  - In this paper, we show how multiple existing QA datasets can be used to evaluate MRS by requiring an open-domain system to perform well on all of them at once.
  - In the following we describe our system DrQA for MRS which consists of two components: 
    - (1) the Document Retriever module for finding relevant articles and 
    - (2) a machine comprehension model, Document Reader, for extracting answers from a single document or a small collection of documents.
    - Retriever + Reader 架构至少在2017年已经提出
  - Document Retriever
    - we use an efficient (non-machine learning) document retrieval system to first narrow our search space and focus on reading only articles that are likely to be relevant. 
  - Document Reader
    - Our Document Reader model is inspired by the recent success of neural network models on machine comprehension tasks, in a similar spirit to the AttentiveReader described in (Hermann et al., 2015; Chen et al., 2016).
    - RNN model predicting the two ends of the span. 
  - Experimental Setup:
    - (WebQuestions 2013(WQ), CuratedTREC 2015(TREC), WikiMovies 2016, SQuAD v1.1 2016)
    - Evidence Corpus
      - We use the 2016-12-21 dump of English Wikipedia for all of our full-scale experiments as the knowledge source used to answer questions.
- Sat, 1 Jun 2019 [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/abs/1906.00300)
  - 将 ORQA 列在历史里，而将 DPR 列为 Retrieval 模型的第一篇是否有失偏颇？
  - open domain question answering (QA)
    - Due to recent advances in reading comprehension systems, 
    - there has been a revival of interest in open domain question answering (QA), 
    - where the evidence must be retrieved from an open corpus, rather than being given as input. 
    - This presents a more realistic scenario for practical applications.
  - However, QA is fundamentally different from IR (Singh, 2012).
  - Retriever component
    - Query 和 Document使用不同模型，也就是要训练两个模型，inner product 作为相似函数。Embeddings 投影到维度 128。
    - Retriever模型结构跟DPR大差不差
  - Reader component 
    - The reader is a span-based variant of the reading comprehension model proposed in Devlin et al. (2018):
    - Reader模型结构跟DPR大差不差
  - Inverse Cloze Task
    - Since this is impractical to learn from scratch, we pre-train the retriever with an Inverse Cloze Task.
    - 虽然Retriever component模型结构和DPR大差不差，差异在 loss，DPR 使用 Metric Learning 方法。
  - Experimental Setup:
    - Evidence Corpus
      - We use the English Wikipedia snapshot from December 20, 2018 as the evidence corpus.
      - The corpus is greedily split into chunks of at most 288 wordpieces based on BERT’s tokenizer, while preserving sentence boundaries.
    - We train and evaluate on data from 5 existing question answering or reading comprehension datasets.
    - (Natural Questions 2019(NQ), WebQuestions 2013(WQ), CuratedTREC 2015(TREC), TriviaQA 2017, SQuAD v1.1 2016) 与 DPR 相同
    - we convert them to open formats, following DrQA (Chen et al., 2017).
      - Natural Questions 
        - we only keep questions with short answers and discard the given evidence document. 
        - Answers with many tokens often resemble extractive snippets rather than canonical answers, so we discard answers with more than 5 tokens.
      - WebQuestions
        - The answers are annotated with respect to Freebase, but we only keep the string representation of the entities.
      - CuratedTrec
      - TriviaQA
        - We use their unfiltered set and discard their distantly supervised evidence
      - SQuAD
    - | Dataset           | Train | Dev  | Test  |
      |-------------------|-------|------|-------|
      | Natural Questions | 79168 | 8757 | 3610  |
      | WebQuestions      | 3417  | 361  | 2032  |
      | CuratedTrec       | 1353  | 133  | 694   |
      | TriviaQA          | 78785 | 8837 | 11313 |
      | SQuAD             | 78713 | 8886 | 10570 |
  - Main Results
    - BM25 + BERT 在 TriviaQA SQuAD 效果好
    - ORQA(ours) 在 Natural Questions, WebQuestions, CuratedTrec 上效果好
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
- Tue, 27 Aug 2019 [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
  - sbert 是 Siamese Dual Encoder 也就是一个模型，DPR 是 Asymmetric Dual Encoder 也就是两个模型。SDE效果比ADE好，当然这是后话了。
  - 这篇论文没有向open domain question answering 方向发力，非常可惜，评估方法现在没什么论文follow，所以一般都没有跟DPR进行比较。当然sentence-transformers的名气也很大了。
  - [sentence-transformers](https://github.com/UKPLab/sentence-transformers/)
  - [Document](https://www.sbert.net/)
  - Ablation Study
    - Pooling Strategy MEAN > CLS > MAX
- Fri, 10 Apr 2020 [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
  - DPR 论文提出的整个体系，包括模型、训练、在线推理，都跟现在主流相近，数据集的处理方式最新论文还在follow，在几个数据集上的结果最新论文仍作为baseline比较。那就多写一点。
  - Transformer 2017年发布，BERT 2019年发布，开始刷nlp任务。真是勃勃生机万物竞发的时代。
  - 使用BERT预训练模型，Embeddings 维度 768，Query 和 Document使用不同模型，也就是要训练两个模型，inner product 作为相似函数， loss function 使用 negative log likelihood
  - DPR performs consistently better than BM25 on all datasets. （Dense Retrieval 登上历史的舞台
  - Experimental Setup:
    - Evidence Corpus
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
  - Main Results
    - DPR 全面超越 BM25，multi-dataset训练出来的模型效果更好，DPR+BM25相互补充效果稍微有提高
    - <img src="https://github.com/noooop/noooop.github.io/blob/main/applications/rag/dpr2.png?raw=true" width="400">
    - <img src="https://github.com/noooop/noooop.github.io/blob/main/applications/rag/dpr1.png?raw=true" width="400">
- Sun, 18 Apr 2021 [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)
  - We first describe an unsupervised approach, which takes an input sentence and predicts itself in a contrastive objective, 
  - with only standard dropout used as noise. 
  - This simple method works surprisingly well, performing on par with previous supervised counterparts. 
  - We find that dropout acts as minimal data augmentation, and removing it leads to a representation collapse.
    - 有意思。 但 unsupervised 效果甚至不如 BM25
- Thu, 19 Aug 2021 [Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models](https://arxiv.org/abs/2108.08877)
  - Google 的 sentence embeddings from text-to-text transformers (ST5)
    - ST5-Enc Base 110M, Large 335M, 3B 1.24B, 11B 4.8B
    - ST5-EncDec Base 110M, Large 335M, 3B 3B, 11B 11B
  - ST5-Enc mean 效果比 ST5-EncDec first 和 ST5-Enc first 效果好。
  - encoder-only 对于 Retrieval 任务已经足够了
- Wed, 15 Dec 2021 [Large Dual Encoders Are Generalizable Retrievers](https://arxiv.org/abs/2112.07899)
  - Google 的 Generalizable T5-based dense Retrievers (GTR)
  - Base 110M, Large 335M, XL 1.24B, XXL 4.8B
- Thu, 14 Apr 2022 [Exploring Dual Encoder Architectures for Question Answering](https://arxiv.org/abs/2204.07120)
  - Dual encoders have been used for questionanswering (QA) and information retrieval (IR) tasks with good results.
  - Previous research focuses on two major types of dual encoders,
    - Siamese Dual Encoder (SDE), with parameters shared across two encoders,  (SBERT (Reimers and Gurevych, 2019), ST5 (Ni et al., 2021b) 
    - and Asymmetric Dual Encoder (ADE), with two distinctly parameterized encoders. (DPR (Karpukhin et al., 2020), DensePhrases (Lee et al., 2021a) 
  - we show that SDE performs significantly better than ADE.
  - We further propose three different improved versions of ADEs by sharing or freezing parts of the architectures between two encoder towers.
    - We find that sharing parameters in projection layers would enable ADEs to perform competitively with or outperform SDEs.
    - We further explore and explain why parameter sharing in projection layer significantly improves the efficacy of the dual encoders, by directly probing the embedding spaces of the two encoder towers with t-SNE algorithm.
  - Main Results
    - By directly probing the embedding space, we demonstrate that the shared projection layers in SDE and ADE-SPL maps the embeddings of the two encoder towers into coinciding parameter spaces, 
    - which is crucial for improving the retrieval quality. Therefore, we recommend to share the projection layers between two encoders of ADEs in practice.
    - <img src="https://github.com/noooop/noooop.github.io/blob/main/applications/rag/edea.png?raw=true" width="400">
    - 这个结论可以泛化在整个在 Metric Learning 问题
- Thu, 26 May 2022 [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
  - 支持多个向量维度
- Thu, 13 Oct 2022 [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)
  - Hugging Face 主导的 Massive Text Embedding Benchmark
  - https://huggingface.co/spaces/mteb/leaderboard
- Sun, 27 Nov 2022 [Dense Text Retrieval based on Pretrained Language Models: A Survey](https://arxiv.org/abs/2211.14876)
  - 2022 年对于 Dense Text Retrieval 的 Survey 已经有 351 引用
  - 其中包括 6 篇 之前的 Survey。行吧
- Wed, 7 Dec 2022 [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533)
  - 微软的E5
  - We pre-train on our proposed text pair dataset for three model sizes: E5small, E5base and E5large initialized from MiniLM, bert-base-uncased, and bert-large-uncased-whole-wordmasking respectively
  - E5-PTlarge 也就略好于 BM25，Weakly-Supervised Contrastive Pre-training 不如 Supervised models
  - Supervised models E5large 比之前 GTRxxl、Sentence-T5xxl强
  - Weakly-Supervised Contrastive Pre-training + Supervised Fine-tuning 称为 sota 模型的标配
- Sun, 31 Dec 2023 [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)
  - LLM as Retrieval +1
- Sun, 4 Feb 2024 [为RAG而生-BCE embedding技术报告](https://zhuanlan.zhihu.com/p/681370855)
  - 二阶段检索器（Two-stage Retriever）“离线”的Embedding搭配“在线”的Reranker
  - 难负样例挖掘？
    - 我们在训练Embedding模型时发现，过难的负样本对模型训练有损害，训练过程中会使模型“困惑”，[影响模型最终性能](https://kexue.fm/archives/8847#%E9%9A%BE%E6%90%9E%E7%9A%84%E9%98%88%E5%80%BC)。
    - 在大量的语料库中，脱离人工校验的自动化难负样例挖掘，难免会“挖到正例”。
    - 其实所谓的“正例”和“难负样例”是根据你业务的定义来的。
    - 所以回归业务目标和好的检索器的“评判标准”，Embedding模型应该能尽量召回相关片段，不要将Reranker要干的精排任务强压在Embedding身上，“越俎代庖”终究会害了它。
- Mon, 5 Feb 2024 [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216)
  - BAAI的BGE M3
  - we introduce a new embedding model called M3-Embedding Supervised models
    - Multi-Linguality:  It provides a uniform support for the semantic retrieval of more than 100 working languages. Enables both multilingual retrieval within each language and crosslingual retrieval between different languages.
    - Multi-Functionality: It can simultaneously accomplish the three common retrieval functionalities: dense retrieval, multi-vector retrieval, and sparse retrieval.
    - Multi-Granularity:  Besides, it is also capable of processing inputs of different granularities, spanning from short sentences to long documents of up to 8,192 tokens
  - Related Work
    - powerful text encoders bert 2019, DPR 2020, ST5 2022
    - negative sampling (Xiong et al., 2020; Qu et al.,2021) 
    - knowledge distillation (Hofstatter et al. ¨ , 2021; Ren et al., 2021; Zhang et al., 2021a). 
    - 之前的 Dense Retrieval：Contriever (Izacard et al., 2022), LLM-Embedder (Zhang et al., 2023a), E5 (Wang et al., 2022), BGE (Xiao et al., 2023), SGPT (Muennighoff, 2022), and Open Text Embedding (Neelakantan et al., 2022),
  - In our work, the following technical contributions are made to optimize the embedding quality. 
    - Firstly, we propose a novel self knowledge distillation framework
      - the [CLS] embedding is used for dense retrieval, while embeddings from other tokens are used for sparse retrieval and multi-vector retrieval.
      - we integrate the relevance scores from different retrieval functions as the teacher signal, which is used to enhance the learning process via knowledge distillation.
    - Secondly, we optimize the batching strategy to achieve a large batch size and high training throughput, which substantially contributes to the discriminativeness of embeddings. 
    - Last but not least, we perform extensive and high-quality data curation. 
      - Our dataset includes three sources: 
        - 1) the extraction of unsupervised data from massive multi-lingual corpora, In total, it brings in 1.2 billion text pairs of 194 languages and 2655 cross-lingual correspondences.
        - 2) we collect relatively small but diverse and high-quality fine-tuning data from labeled corpora. we incorporate 8 datasets, For Chinese, we integrate 7 datasets, For other languages, we leverage the training data from Mr. Tydi (Zhang et al., 2021b) and MIRACL (Zhang et al., 2023c).
        - 3) the synthesization of scarce training data
          - Specifically, we sample lengthy articles from Wikipedia, Wudao (Yuan et al., 2021) and mC4 datasets and randomly choose paragraphs from them. 
          - Then we use GPT3.5 to generate questions based on these paragraphs.
          - 使用 GPT 合成数据训练模型开始成为主流
      - The three data sources are complement to each other and applied to different training stages, which lays a solid foundation for the versatile text embeddings.
    - Train
      - loss
        - minimize the InfoNCE loss(NCE stands for Noise-Contrastive Estimation)
      - native multi-objective training can be unfavorable to the embedding’s quality.
        -  we integrate the relevance scores from different retrieval functions as the teacher signal, which is used to enhance the learning process via knowledge distillation.
      - The training process constituta multi-stage workflow
        - the text encoder (an XLM-RoBERTa (Conneau et al., 2020) model adapted by RetroMAE (Xiao et al., 2022) method) is pre-trained with the massive unsupervised data, where only the dense retrieval is trained in the basic form of contrastive learning.
        - The self-knowledge distillation is applied to the second stage, where the embedding model is fine-tuned to establish the three retrieval functionalities.
          - Both labeled and synthetic data are used in this stage, where hard negative samples are introduced for each query following the ANCE method (Xiong et al., 2020).
      - Efficient Batch
        - It also needs to keep the batch size as large as possible(introducing a huge amount of in-batch negatives) to ensure the discriminativeness of text embeddings
        - Particularly, the training data is pre-processed by being grouped by sequence length. When producing a mini-batch, the training instances are sampled from the same group.
        - We iteratively encode each sub-batch using gradient checkpointing (Chen et al., 2016)and gather all generated embeddings.
        - Finally, the embeddings from different GPUs are broadcasted, allowing each device to obtain all embeddings in the distributed environment, 
          - which notably expands the scale of in-bath negative samples.
    - Experiment
      - Multi-Lingual Retrieval
      - Cross-Lingual Retrieval
      - Multilingual Long-Doc Retrieval
    - Ablation study
      - Self-knowledge distillation
      - Impact of multi-stage training
         - | Model (Dense)                | MIRACL | 
           |------------------------------|--------| 
           | Fine-tune                    | 60.5   | 
           | RetroMAE + Fine-tune         | 66.1   | 
           | RetroMAE + Unsup + Fine-tune | 69.2   |
- Sat, 24 Feb 2024 [OpenAI vs Open-Source Multilingual Embedding Models Choosing the model that works best for your data](https://towardsdatascience.com/openai-vs-open-source-multilingual-embedding-models-e5ccb7c90f05)
  - Generate a custom Q/A dataset
  - 介绍了一种用ChatGPT合成问答数据集测试检索模型的方法
  - 用ChatGPT合成问答数据集训练模型，用ChatGPT合成问答数据集测试模型的世界达成了
- Wed, 27 Mar 2024 [Scaling Laws For Dense Retrieval](https://arxiv.org/abs/2403.18684)
  - 24 BERT checkpoints from the original Google release, with model sizes ranging from 0.5 million (BERT-Tiny) to 82 million parameters (BERT-Base)
  - For experiments on Chinese retrieval benchmarks, we selected the ERNIE series
  - <img src="https://github.com/noooop/noooop.github.io/blob/main/applications/rag/sldr.png?raw=true" width="400">
- Tue, 9 Apr 2024 [LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders](https://arxiv.org/abs/2404.05961)
  - additional training phase with a specially designed masked token prediction to warm-up the bidirectional attention.
  - LLM as Retrieval +2
- Wed, 8 May 2024 [Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models](https://arxiv.org/abs/2405.05374)
  - we conduct two training rounds using two different kinds of datasets.
    - The initial training round is large-scale pretraining using only in-batch negative examples.
    - The second round of training (often referred to as the fine-tuning step) calls for similar pairs of queries and documents augmented with an additional set of “hard” negative documents
  - How hard should these negatives be for maximally effective learning in the fine-tuning phase? 
    - Our answer to this question was ultimately a tunable hard negative mining strategy in which we leveraged a preexisting text embedding model to identify and score the hardest negatives for each training example. 
    - Then, we applied a score threshold to discard the hard negatives from the above set. 
    - We found that using an upper threshold rather than a specific rank helped account for the fact that some queries admit much harder top-k negatives than others.
    - we perform a parameter sweep of the negative hardness threshold to demonstrate the value of a tunable approach (the optimal threshold value scores significantly better than other choices). 
- Sat, 11 May 2024 [Piccolo2: General Text Embedding with Multi-task Hybrid Loss Training](https://arxiv.org/abs/2405.06932)
  - Multi-task Hybrid Loss
- Mon, 27 May 2024 [NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models](https://arxiv.org/abs/2405.17428)
  - LLM as Retrieval +3
- Mon, 27 May 2024 [Recent advances in text embedding: A Comprehensive Review of Top-Performing Methods on the MTEB Benchmark](https://arxiv.org/abs/2406.01607)
- Mon, 22 Jul 2024 [NV-Retriever: Improving text embedding models with effective hard-negative mining](https://arxiv.org/abs/2407.15831)
  - hard-negative mining
- Fri, 26 Jul 2024 [bge-multilingual-gemma2,bge-en-icl](https://github.com/FlagOpen/FlagEmbedding/tree/master)
  - 2024-07-31 MTEB en榜单 1.bge-en-icl 2.stella_en_1.5B_v5 3.SFR-Embedding-2_R 4.gte-Qwen2-7B-instruct 5.stella_en_400M_v5 6.bge-multilingual-gemma2 7.NV-Embed-v1 8.voyage-large-2-instruct 9.Linq-Embed-Mistral 10.SFR-Embedding-Mistral  
  - LLM as Retrieval +4 +5
- Mon, 29 Jul 2024 [mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval](https://arxiv.org/abs/2407.19669)
  - 从头训练一个基础模型，并微调成一个 Retrieval(Embeddings) model 和 Reranking Model, 有钱真好
  - pre-train
    - We pre-train the model via masked language modeling (MLM)，The MLM probability is set to 30% 
    - To train the native 8192-context model more efficiently, we adopt a phased training curriculum (Xiong et al., 2024)
      - MLM-2048: we chunk the input into 2048 tokens and set RoPE base to 10, 000.
      - MLM-8192: we chunk the input into 8192 tokens and set RoPE base to 160, 000.
  - Retrieval(Embeddings) model
    - we construct the TRM for first-stage text retrieval in two steps: 
      - contrastive pre-training and fine-tuning (Wang et al., 2022; Li et al., 2023). 
      - Both steps share the same InfoNCE(Oord et al., 2018) learning objective
    - Contrastive Pre-Training
    - Matryoshka Embedding
    - Sparse Representation
    - Contrastive Fine-Tuning 
  - Text Reranking Model
    - It takes the query and document as input: [CLS] q [SEP] d, and directly predicts their relevance score by the [CLS] output state:
    - srerank = W h[CLS]
    - The model is fine-tuned by InfoNCE in one step6 based on our text encoder
  - 论文没有提 Hard Example Mining，不知道是想表达 no bells and whistles，[Stella_v5](https://github.com/DunZhang/Stella/blob/main/news_and_todo.md)系列在这个基础上微调效果就好一些。
  - [gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) [gte-Qwen1.5-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-7B-instruct) 
    - gte-Qwen2-7B-instruct is the latest model in the gte (General Text Embedding) model family that ranks No.1 in both English and Chinese evaluations on the Massive Text Embedding Benchmark MTEB benchmark (as of June 16, 2024).
    - LLM as Retrieval +6 +7  居然没有写在论文里
  
# 总结，如何训练一个效果很好的Retrieval(Embeddings) model
- 模型
  - Dense Retrieval 总体展现出明显的 Scaling Laws，但时不时也有小模型的在MTEB榜单前列
  - 选择适合的基础模型，多语言能力和长上下文能力比较重要，但 bert 都是2020年左右训练的，普遍不如现在llm训练的充分
  - 越来越多的 Large decoder-only language models (LLMs) as Retrieval的模型上MTEB榜，基础模型选择范围就大大拓宽了
  - 更有钱的公司会从头训练一个基础模型，微调成一个检索模型 比如 ST5、mGTE。。
- 数据
  - 使用 GPT 合成数据
  - Weakly-Supervised Contrastive Pre-training + Supervised Fine-tuning
- 算法
  - 适合难度的 Hard Example Mining
  - 从其他 Metric Learning 和 Contrastive Learning 学习任务中寻找启发
  - 多任务学习，模型蒸馏

# Rerank model
很多文章吧 Rerank model 称为 cross-encoder，相对与 Dense Retrieval 的 dual-encoder。
相比 Dense Retrieval 模型，算法上可以结合Metric Learning 和 Contrastive Learning，系统上可以跟 approximate nearest neighbor 结合， 下游任务又可以跟  large-scale open-domain QA 结合。
Rerank model 真的要无聊很多，Rerank model 本质上就是个二分类任务，约等于 bert 预训练任务 Next Sentence Prediction。
- Sun, 4 Feb 2024 [为RAG而生-BCE embedding技术报告](https://zhuanlan.zhihu.com/p/681370855)
  - 我们将BCEmbedding设计为二阶段检索器，分工合作：“离线”的Embedding负责尽可能召回，“在线”的Reranker负责精排和低质量过滤。
    - 精排阶段为了解决信息交互的问题，采用cross-encoder架构（如图二-2 (b)所示）。Reranker模型可以实现用户问题和知识库语料的信息交互，使模型可以“见机行事”地识别到更加准确的语义关系，算法性能上限可以很高。该方式的缺点是，需要对用户问题和知识库语料进行在线（online）地语义关系提取，效率比较低，无法对全量的知识库语料进行实时处理。
    - 结合召回和精排二者的优势，召回阶段可以快速找到用户问题相关文本片段，精排阶段可以将正确相关片段尽可能排在靠前位置，并过滤掉低质量的片段。二阶段检索可以很好地权衡检索效果和效率，具有巨大应用价值。
  - 有意义的Rerank分数
    - “评判标准”中好检索器还有一个特点，可以过滤低质量信息。我们设计的Reranker模型，输出的(query, passage)语义相关分数，不仅能用来做psaages排序，其分数的绝对值可表征真实的语义相关程度，这可以用来判断哪些是低质量passages，实现低质量片段过滤。这对RAG中LLM回答问题非常有帮助，更干练、干扰信息少的context，可以有效提高LLM回答质量[17]。
    - 根据我们业务实践经验和开源社区的反馈，bce-reranker-base_v1输出的分数推荐以0.35～0.4为阈值，来进行低质量passage过滤。用户实际使用反馈，收获很不错的效果。
- Mon, 18 Mar 2024 [bge-reranker-v2-m3、BAAI/bge-reranker-v2-gemma、 BAAI/bge-reranker-v2-minicpm-layerwise.](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker)
  - reranker 都不配有篇技术报告
  - LLM-based reranker
- Tue, 25 Jun 2024 [Jina Reranker v2 for Agentic RAG: Ultra-Fast, Multilingual, Function-Calling & Code Search](https://jina.ai/news/jina-reranker-v2-for-agentic-rag-ultra-fast-multilingual-function-calling-and-code-search)
  - Jina Reranker v2 将 Reranker 玩出了新高度
  - Multilingual: More relevant search results in 100+ languages, outperforming bge-reranker-v2-m3;
  - Agentic: State-of-the-art function-calling and text-to-SQL aware document reranking for agentic RAG;
  - Code retrieval: Top performance on code retrieval tasks, and
  - Ultra-fast: 15x more documents throughput than bge-reranker-v2-m3, and 6x more than jina-reranker-v1-base-en.
- Fri, 26 Jul 2024 [bge-reranker-v2.5-gemma2-lightweight](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight)
  - trained based on gemma2-9b
- Mon, 29 Jul 2024 [mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval](https://arxiv.org/abs/2407.19669)
  - 从头训练一个基础模型，并微调成一个 Retrieval(Embeddings) model 和 Reranking Model, 有钱真好

# learned sparse representations \ late-interaction methods
- Mon, 27 Apr 2020 [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
- Mon, 5 Feb 2024 [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216)
- Mon, 23 Sep 2024 [Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling](https://arxiv.org/abs/2409.14683)


# Generation-Augmented Retrieval
你没看错，是生成增强检索
- Thu, 17 Sep 2020 [Generation-Augmented Retrieval for Open-domain Question Answering](https://arxiv.org/abs/2009.08553)
- Tue, 20 Dec 2022 [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
  -  Given a query, HyDE first zero-shot instructs an instruction-following language model (e.g. InstructGPT) to generate a hypothetical document.
- Wed, 31 Jan 2024 [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)

# Chunking / Chucking Granularity
[跳转](./awesome_rag.md#chunking--chucking-granularity)
  

# Toolkit 
- Tue, 27 Aug 2019 [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
  - [sentence-transformers](https://github.com/UKPLab/sentence-transformers/)
  - [Document](https://www.sbert.net/)
- Fri, 19 Feb 2021 [Pyserini: An Easy-to-Use Python Toolkit to Support Replicable IR Research with Sparse and Dense Representations](https://arxiv.org/abs/2102.10073)
  - [pyserini](https://github.com/castorini/pyserini)

# Other
- Fri, 23 Feb 2024 [Self-Retrieval: Building an Information Retrieval System with One Large Language Model](https://arxiv.org/abs/2403.00801)
  - LLM can memorize (passage -> title)
  - 至少Hit@1、 Hit@5、 MRR@5 指标比 dense retrieval 模型 GTR BGE OpenAI 效果好?? 所以 dense retrieval 必须配合 reranker ??
