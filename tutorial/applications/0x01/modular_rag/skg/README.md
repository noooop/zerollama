# Self-Knowledge Guided Retrieval Augmentation for Large Language Models (EMNLP Findings 2023)

<img src="https://github.com/THUNLP-MT/SKR/blob/main/figs/skr.png?raw=true" width="800">

###  TL;DR
提出一种 Adaptive Retrieval Augmentation 方法，将 query 二分类，需不需检索

> (for LLM known)
> 
> A: (LLM directly answers without retrieval)
> (for LLM unknown)
> 
> Here are some passages: {pt}
> 
> A: (LLM answers with retrieval augmentation)

提出四种方法构造分类器
- Direct Prompting，直接问llm
- in-Context Learning，给llm几个例子
- Training a Classifier，比如 BERT
- Nearest Neighbor Search


### Reference
[GITHUB](https://github.com/THUNLP-MT/SKR)

[arxiv](https://arxiv.org/abs/2310.05002)

[paper](https://aclanthology.org/2023.findings-emnlp.691.pdf)