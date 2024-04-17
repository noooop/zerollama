# Hugging Face Transformers

# TL;DR
🤗 Transformers 提供了数以千计的预训练模型，支持 100 多种语言的文本分类、信息抽取、问答、摘要、翻译、文本生成。它的宗旨是让最先进的 NLP 技术人人易用。

🤗 Transformers 提供了便于快速下载和使用的API，让你可以把预训练模型用在给定文本、在你的数据集上微调然后通过 [model hub](https://huggingface.co/models) 与社区共享。同时，每个定义的 Python 模块均完全独立，方便修改和快速研究实验。

🤗 Transformers 支持三个最热门的深度学习库： [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) 以及 [TensorFlow](https://www.tensorflow.org/) — 并与之无缝整合。你可以直接使用一个框架训练你的模型然后用另一个加载和推理。

# 注意
1. 考虑到大语言模型动辄几个G，下载相当缓慢，所以默认设置了local_files_only=True， 模型必须先下载，才能加载。
2. 设置全局的Hugging Face代理， [请参考](https://github.com/noooop/zerollama/tree/main/zerollama/core/config)
3. 设置全局的Hugging Face本地缓存文件目录，[请参考](https://github.com/noooop/zerollama/tree/main/zerollama/core/config)

# Reference
[GITHUB](https://github.com/huggingface/transformers)

[Documentation](https://huggingface.co/docs/transformers/index)


