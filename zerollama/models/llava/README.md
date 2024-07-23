# LLaVA
LLaVA: Large Language and Vision Assistant

##  TL;DR
*Visual instruction tuning towards large language and vision models with GPT-4 level capabilities.*

[📢 [LLaVA-NeXT Blog](https://llava-vl.github.io/blog/2024-01-30-llava-next/)] [[Project Page](https://llava-vl.github.io/)] [[Demo](https://llava.hliu.cc/)]  [[Data](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)] [[Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)]


### News
- [2024/05/10] 🔥 **LLaVA-NeXT** (Stronger) models are released, stronger LMM with support of LLama-3 (8B) and Qwen-1.5 (72B/110B). [[Blog](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)] [[Checkpoints](https://huggingface.co/collections/lmms-lab/llava-next-6623288e2d61edba3ddbf5ff)] [[Demo](https://llava-next.lmms-lab.com/)] [[Code](https://github.com/LLaVA-VL/LLaVA-NeXT/)] 
- [2024/05/10] 🔥 **LLaVA-NeXT** (Video) is released. The image-only-trained LLaVA-NeXT model is surprisingly strong on video tasks with zero-shot modality transfer. DPO training with AI feedback on videos can yield significant improvement. [[Blog](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/)] [[Checkpoints](https://huggingface.co/collections/lmms-lab/llava-next-video-661e86f5e8dabc3ff793c944)] [[Code](https://github.com/LLaVA-VL/LLaVA-NeXT/)]
- [03/10] Releasing **LMMs-Eval**, a highly efficient evaluation pipeline we used when developing LLaVA-NeXT. It supports the evaluation of LMMs on dozens of public datasets and allows new dataset onboarding, making the dev of new LMMs much faster. [[Blog](https://lmms-lab.github.io/lmms-eval-blog/lmms-eval-0.1/)] [[Codebase](https://github.com/EvolvingLMMs-Lab/lmms-eval)]
- [1/30] 🔥 **LLaVA-NeXT** (LLaVA-1.6) is out! With additional scaling to LLaVA-1.5, LLaVA-NeXT-34B outperforms Gemini Pro on some benchmarks. It can now process 4x more pixels and perform more tasks/applications than before. Check out the [blog post](https://llava-vl.github.io/blog/2024-01-30-llava-next/), and explore the [demo](https://llava.hliu.cc/)! Models are available in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). Training/eval data and scripts coming soon.
- [11/10] [LLaVA-Plus](https://llava-vl.github.io/llava-plus/) is released: Learning to Use Tools for Creating Multimodal Agents, with LLaVA-Plus (LLaVA that Plug and Learn to Use Skills). [[Project Page](https://llava-vl.github.io/llava-plus/)] [[Demo](https://llavaplus.ngrok.io/)] [[Code](https://github.com/LLaVA-VL/LLaVA-Plus-Codebase)] [[Paper](https://arxiv.org/abs/2311.05437)]
- [11/2] [LLaVA-Interactive](https://llava-vl.github.io/llava-interactive/) is released: Experience the future of human-AI multimodal interaction with an all-in-one demo for Image Chat, Segmentation, Generation and Editing. [[Project Page](https://llava-vl.github.io/llava-interactive/)] [[Demo](https://llavainteractive.ngrok.io/)] [[Code](https://github.com/LLaVA-VL/LLaVA-Interactive-Demo)] [[Paper](https://arxiv.org/abs/2311.00571)]
- [10/26] 🔥 LLaVA-1.5 with LoRA achieves comparable performance as full-model finetuning, with a reduced GPU RAM requirement ([ckpts](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#llava-v15), [script](https://github.com/haotian-liu/LLaVA#train)). We also provide a [doc](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md) on how to finetune LLaVA-1.5 on your own dataset with LoRA.
- [10/12] Check out the Korean LLaVA (Ko-LLaVA), created by ETRI, who has generously supported our research! [[🤗 Demo](https://huggingface.co/spaces/etri-vilab/Ko-LLaVA)]
- [10/5] 🔥 LLaVA-1.5 is out! Achieving SoTA on 11 benchmarks, with just simple modifications to the original LLaVA, utilizes all public data, completes training in ~1 day on a single 8-A100 node, and surpasses methods like Qwen-VL-Chat that use billion-scale data. Check out the [technical report](https://arxiv.org/abs/2310.03744), and explore the [demo](https://llava.hliu.cc/)! Models are available in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). The training data and scripts of LLaVA-1.5 are released [here](https://github.com/haotian-liu/LLaVA#train), and evaluation scripts are released [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)!
- [9/26] LLaVA is improved with reinforcement learning from human feedback (RLHF) to improve fact grounding and reduce hallucination. Check out the new SFT and RLHF checkpoints at project [[LLavA-RLHF]](https://llava-rlhf.github.io/)
- [9/22] [LLaVA](https://arxiv.org/abs/2304.08485) is accepted by NeurIPS 2023 as **oral presentation**, and [LLaVA-Med](https://arxiv.org/abs/2306.00890) is accepted by NeurIPS 2023 Datasets and Benchmarks Track as **spotlight presentation**.

### License Agreement

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)

**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-2 and Vicuna-v1.5). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.

### Reference
[GITHUB](https://github.com/haotian-liu/LLaVA)

[Hugging Face](https://huggingface.co/liuhaotian/)
