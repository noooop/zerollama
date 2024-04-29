# BAAI
[北京智源研究院](https://www.baai.ac.cn/)

bge is short for BAAI general embedding.

## BGE M3-Embedding

###  TL;DR
In this project, we introduce BGE-M3, which is distinguished for its versatility in Multi-Functionality, Multi-Linguality, and Multi-Granularity. 
- Multi-Functionality: It can simultaneously perform the three common retrieval functionalities of embedding model: dense retrieval, multi-vector retrieval, and sparse retrieval. 
- Multi-Linguality: It can support more than 100 working languages. 
- Multi-Granularity: It is able to process inputs of different granularities, spanning from short sentences to long documents of up to 8192 tokens. 

### requirements
FlagEmbedding==1.2.9

### News
- 2024/3/20: **Thanks Milvus team!** Now you can use hybrid retrieval of bge-m3 in Milvus: [pymilvus/examples
/hello_hybrid_sparse_dense.py](https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py).
- 2024/3/8: **Thanks for the [experimental results](https://towardsdatascience.com/openai-vs-open-source-multilingual-embedding-models-e5ccb7c90f05) from @[Yannael](https://huggingface.co/Yannael). In this benchmark, BGE-M3 achieves top performance in both English and other languages, surpassing models such as OpenAI.**
- 2024/3/2: Release unified fine-tuning [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/unified_finetune) and [data](https://huggingface.co/datasets/Shitao/bge-m3-data) 
- 2024/2/6: We release the [MLDR](https://huggingface.co/datasets/Shitao/MLDR) (a long document retrieval dataset covering 13 languages) and [evaluation pipeline](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB/MLDR). 
- 2024/2/1: **Thanks for the excellent tool from Vespa.** You can easily use multiple modes of BGE-M3 following this [notebook](https://github.com/vespa-engine/pyvespa/blob/master/docs/sphinx/source/examples/mother-of-all-embedding-models-cloud.ipynb)


### License Agreement
FlagEmbedding is licensed under the MIT License.

### Reference
[GITHUB](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3M)

[Hugging Face](https://huggingface.co/BAAI/bge-m3)

[Paper Link](https://arxiv.org/abs/2402.03216)