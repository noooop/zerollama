

# Adaptive Retrieval Augmentation
[Self-Knowledge Guided Retrieval Augmentation for Large Language Models (EMNLP Findings 2023)](https://arxiv.org/abs/2310.05002)
[GITHUB](https://github.com/THUNLP-MT/SKR)
- 提出一种 Adaptive Retrieval Augmentation 方法，将 query 二分类，需不需检索
- 提出四种方法构造分类器
  - Direct Prompting，直接问llm
  - in-Context Learning，给llm几个例子
  - Training a Classifier，比如 BERT
  - Nearest Neighbor Search


[When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories](https://arxiv.org/abs/2212.10511)

# Query Rewriter
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