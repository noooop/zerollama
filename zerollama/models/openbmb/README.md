# OpenBMB
OpenBMB开源社区由清华大学自然语言处理实验室和面壁智能共同支持发起。

## MiniCPM-2B

###  TL;DR
MiniCPM 是面壁智能与清华大学自然语言处理实验室共同开源的系列端侧大模型，主体语言模型 MiniCPM-2B 仅有 24亿（2.4B）的非词嵌入参数量, 总计2.7B参数量。
- 经过 SFT 后，MiniCPM-2B 在公开综合性评测集上与 Mistral-7B 表现相近（中文、数学、代码能力更优），整体性能超越 Llama2-13B、MPT-30B、Falcon-40B 等模型。
- 经过 DPO 后，MiniCPM-2B 在当前最接近用户体感的评测集 MTBench 上也超越了 Llama2-70B-Chat、Vicuna-33B、Mistral-7B-Instruct-v0.1、Zephyr-7B-alpha 等众多代表性开源大模型。
- 以 MiniCPM-2B 为基础构建端侧多模态大模型 MiniCPM-V 2.0，在多个测试基准中实现了 7B 以下模型的最佳性能，在 OpenCompass 榜单上超过了 Qwen-VL-Chat 9.6B、CogVLM-Chat 17.4B 和 Yi-VL 34B 等更大参数规模的模型。MiniCPM-V 2.0 还展现出领先的 OCR 能力，在场景文字识别能力上接近 Gemini Pro。
- 经过 Int4 量化后，MiniCPM 可在手机上进行部署推理，流式输出速度略高于人类说话速度。MiniCPM-V 也直接跑通了多模态大模型在手机上的部署。
- 一张1080/2080可高效参数微调，一张3090/4090可全参数微调，一台机器可持续训练 MiniCPM，二次开发成本较低。

### News
- 2024/04/11 开源[MiniCPM-V-2.0](https://huggingface.co/openbmb/MiniCPM-V-2.0)、[MiniCPM-2B-128k](https://huggingface.co/openbmb/MiniCPM-2B-128k)、[MiniCPM-MoE-8x2B](https://huggingface.co/openbmb/MiniCPM-MoE-8x2B)和[MiniCPM-1B](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16)！点击[这里](https://openbmb.vercel.app/?category=Chinese+Blog)查看技术博客。** 
- 2024/03/16 MiniCPM-2B 的30余个中间检查点开放了！[HuggingFace链接](https://huggingface.co/openbmb/MiniCPM-2B-history)
- 2024/02/08 我们更新了[llama-format的模型权重](#llamaformat)，方便大家更加快捷地使用我们的模型。
- 2024/02/01 初始发布。

### License 
本仓库中代码依照 Apache-2.0 协议开源

MiniCPM 模型权重的使用则需要遵循 “通用模型许可协议-来源说明-宣传限制-商业授权”。

MiniCPM 模型权重对学术研究完全开放。

如需将模型用于商业用途，请联系cpm@modelbest.cn来获取书面授权，在登记后亦允许免费商业使用。

This repository is released under the Apache-2.0 License.

The usage of MiniCPM model weights must strictly follow the General Model License (GML).

The models and weights of MiniCPM are completely free for academic research.

If you intend to utilize the model for commercial purposes, please reach out to cpm@modelbest.cn to obtain the certificate of authorization.

### Todo
卸载第一个模型没清空缓存，加载第二个模型不成功

```
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |   5624 MiB |   5673 MiB |  22536 MiB |  16911 MiB |
|       from large pool |   5543 MiB |   5589 MiB |   8875 MiB |   3332 MiB |
|       from small pool |     81 MiB |    243 MiB |  13660 MiB |  13579 MiB |
|---------------------------------------------------------------------------|
| Active memory         |   5624 MiB |   5673 MiB |  22536 MiB |  16911 MiB |
|       from large pool |   5543 MiB |   5589 MiB |   8875 MiB |   3332 MiB |
|       from small pool |     81 MiB |    243 MiB |  13660 MiB |  13579 MiB |
|---------------------------------------------------------------------------|
| Requested memory      |   5434 MiB |   5464 MiB |  20916 MiB |  15481 MiB |
|       from large pool |   5353 MiB |   5380 MiB |   7310 MiB |   1957 MiB |
|       from small pool |     81 MiB |    243 MiB |  13606 MiB |  13524 MiB |
|---------------------------------------------------------------------------|
| GPU reserved memory   |   5684 MiB |   5868 MiB |   5870 MiB | 190464 KiB |
|       from large pool |   5600 MiB |   5620 MiB |   5620 MiB |  20480 KiB |
|       from small pool |     84 MiB |    248 MiB |    250 MiB | 169984 KiB |
|---------------------------------------------------------------------------|
| Non-releasable memory |  60505 KiB | 376319 KiB |  20059 MiB |  20000 MiB |
|       from large pool |  58240 KiB | 331208 KiB |   3647 MiB |   3590 MiB |
|       from small pool |   2265 KiB |  56959 KiB |  16412 MiB |  16409 MiB |
|---------------------------------------------------------------------------|
| Allocations           |     629    |     670    |  487593    |  486964    |
|       from large pool |     418    |     443    |    2195    |    1777    |
|       from small pool |     211    |     388    |  485398    |  485187    |
|---------------------------------------------------------------------------|
| Active allocs         |     629    |     670    |  487593    |  486964    |
|       from large pool |     418    |     443    |    2195    |    1777    |
|       from small pool |     211    |     388    |  485398    |  485187    |
|---------------------------------------------------------------------------|
| GPU reserved segments |     324    |     407    |     408    |      84    |
|       from large pool |     282    |     283    |     283    |       1    |
|       from small pool |      42    |     124    |     125    |      83    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |      32    |     256    |  222866    |  222834    |
|       from large pool |      25    |     162    |    1917    |    1892    |
|       from small pool |       7    |      95    |  220949    |  220942    |
|---------------------------------------------------------------------------|
| Oversize allocations  |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize GPU segments |       0    |       0    |       0    |       0    |
|===========================================================================|
5529.2880859375
================================================================================
WARNING:root:Some parameters are on the meta device device because they were offloaded to the cpu.
WARNING:accelerate.big_modeling:You shouldn't move a model that is dispatched using accelerate hooks.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/mnt/e/PycharmProjects/zerollama/zerollama/models/openbmb/minicpm.py", line 58, in <module>
    run(model_name)
  File "/mnt/e/PycharmProjects/zerollama/zerollama/models/openbmb/minicpm.py", line 43, in run
    model.load()
  File "/mnt/e/PycharmProjects/zerollama/zerollama/inference_backend/hf_transformers/main.py", line 66, in load
    self.model = model.to(self.device)
                 ^^^^^^^^^^^^^^^^^^^^^
  File "/root/anaconda3/lib/python3.11/site-packages/accelerate/big_modeling.py", line 455, in wrapper
    raise RuntimeError("You can't move a model that has some modules offloaded to cpu or disk.")
RuntimeError: You can't move a model that has some modules offloaded to cpu or disk.
```

### Reference
[openbmb](https://www.openbmb.cn/home)

[MiniCPM 技术报告](https://shengdinghu.notion.site/MiniCPM-c805a17c5c8046398914e47f0542095a)

[Technical Report ](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20?pvs=4)

[GITHUB](https://github.com/OpenBMB/MiniCPM)

[Hugging Face](https://huggingface.co/openbmb)

