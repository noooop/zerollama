# Hunyuan
腾讯混元

## Hunyuan-DiT

A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding

###  TL;DR
This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for our paper exploring Hunyuan-DiT. You can find more visualizations on our [project page](https://dit.hunyuan.tencent.com/).

> [**Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding**](https://arxiv.org/abs/2405.08748) <br>

> [**DialogGen: Multi-modal Interactive Dialogue System for Multi-turn Text-to-Image Generation**](https://arxiv.org/abs/2403.08857) <br>


### News
* Jul 03, 2024: :tada: Kohya-hydit version now available for v1.1 and v1.2 models, with GUI for inference. Official Kohya version is under review. See [kohya](./kohya_ss-hydit) for details.
* Jun 27, 2024: :art: Hunyuan-Captioner is released, providing fine-grained caption for training data. See [mllm](./mllm) for details.
* Jun 27, 2024: :tada: Support LoRa and ControlNet in diffusers. See [diffusers](./diffusers) for details.
* Jun 27, 2024: :tada: 6GB GPU VRAM Inference scripts are released. See [lite](./lite) for details.
* Jun 19, 2024: :tada: ControlNet is released, supporting canny, pose and depth control. See [training/inference codes](#controlnet) for details.
* Jun 13, 2024: :zap: HYDiT-v1.1 version is released, which mitigates the issue of image oversaturation and alleviates the watermark issue. Please check [HunyuanDiT-v1.1 ](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1) and 
[Distillation-v1.1](https://huggingface.co/Tencent-Hunyuan/Distillation-v1.1) for more details.
* Jun 13, 2024: :truck: The training code is released, offering [full-parameter training](#full-parameter-training) and [LoRA training](#lora).
* Jun 06, 2024: :tada: Hunyuan-DiT is now available in ComfyUI. Please check [ComfyUI](#using-comfyui) for more details.
* Jun 06, 2024: 🚀 We introduce Distillation version for Hunyuan-DiT acceleration, which achieves **50%** acceleration on NVIDIA GPUs. Please check [Distillation](https://huggingface.co/Tencent-Hunyuan/Distillation) for more details.
* Jun 05, 2024: 🤗 Hunyuan-DiT is now available in 🤗 Diffusers! Please check the [example](#using--diffusers) below.
* Jun 04, 2024: :globe_with_meridians: Support Tencent Cloud links to download the pretrained models! Please check the [links](#-download-pretrained-models) below.
* May 22, 2024: 🚀 We introduce TensorRT version for Hunyuan-DiT acceleration, which achieves **47%** acceleration on NVIDIA GPUs. Please check [TensorRT-libs](https://huggingface.co/Tencent-Hunyuan/TensorRT-libs) for instructions.
* May 22, 2024: 💬 We support demo running multi-turn text2image generation now. Please check the [script](#using-gradio) below.


### License Agreement

[TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT](https://github.com/Tencent/HunyuanDiT/tree/main?tab=License-1-ov-file#readme)


### Reference
[GITHUB](https://github.com/Tencent/HunyuanDiT/)

[Hugging Face](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT)
