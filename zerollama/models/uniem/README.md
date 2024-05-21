# M3E
M3E 是 Moka Massive Mixed Embedding 的缩写

##  TL;DR
- Moka，此模型由 MokaAI 训练，开源和评测，训练脚本使用 [uniem](https://github.com/wangyuxinwhy/uniem/blob/main/scripts/train_m3e.py) ，评测 BenchMark 使用 [MTEB-zh](https://github.com/wangyuxinwhy/uniem/tree/main/mteb-zh)
- Massive，此模型通过**千万级** (2200w+) 的中文句对数据集进行训练
- Mixed，此模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索
- Embedding，此模型是文本嵌入模型，可以将自然语言转换成稠密的向量

## News
- 2023.06.24，添加微调 M3E 的教程 [notebook](https://github.com/wangyuxinwhy/uniem/blob/main/examples/finetune.ipynb)，几行代码，更佳适配！<a target="_blank" href="https://colab.research.google.com/github/wangyuxinwhy/uniem/blob/main/examples/finetune.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
- 2023.06.14，添加了三个中文开源文本嵌入模型到评测中，包括 UER, ErLangShen, DMetaSoul
- 2023.06.08，添加检索任务的评测结果，在 T2Ranking 1W 中文数据集上，m3e-base 在 ndcg@10 上达到了 0.8004，超过了 openai-ada-002 的 0.7786
- 2023.06.07，添加文本分类任务的评测结果，在 6 种文本分类数据集上，m3e-base 在 accuracy 上达到了 0.6157，超过了 openai-ada-002 的 0.5956

## License Agreement

M3E models 使用的数据集中包括大量非商用的数据集，所以 M3E models 也是非商用的，仅供研究使用。不过我们已经在 M3E 数据集上标识了商用和非商用的数据集，您可以根据自己的需求自行训练。

## Reference
[GITHUB](https://github.com/wangyuxinwhy/uniem)

[Hugging Face](https://huggingface.co/moka-ai)

