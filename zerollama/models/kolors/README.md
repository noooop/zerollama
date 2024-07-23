## Kolors

Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis

###  TL;DR

可图大模型是由快手可图团队开发的基于潜在扩散的大规模文本到图像生成模型。Kolors 在数十亿图文对下进行训练，在视觉质量、复杂语义理解、文字生成（中英文字符）等方面，相比于开源/闭源模型，都展示出了巨大的优势。同时，Kolors 支持中英双语，在中文特色内容理解方面更具竞争力。更多的实验结果和细节请查看我们的<a href="https://github.com/Kwai-Kolors/Kolors/blob/master/imgs/Kolors_paper.pdf">技术报告</a></b>。

### News
* 2024.07.12 🤗 Kolors is now available in **Diffusers**! Please check [kolors-diffusers](https://huggingface.co/Kwai-Kolors/Kolors-diffusers) or the [example](#using-with-diffusers) below for detail! Thanks to the Diffusers team for their technical support.
* 2024.07.10 🤖 Kolors supports [ModelScope](https://modelscope.cn/models/Kwai-Kolors/Kolors).
* 2024.07.09 💥 Kolors supports [ComfyUI](https://github.com/comfyanonymous/ComfyUI#manual-install-windows-linux). Thanks to [@kijai](https://github.com/kijai/ComfyUI-KwaiKolorsWrapper) with his great work.
* 2024.07.06 🔥🔥🔥 We release **Kolors**, a large text-to-image model trained on billions of text-image pairs. This model is bilingual in both Chinese and English, and supports a context length of 256 tokens. For more technical details, please refer to [technical report](https://github.com/Kwai-Kolors/Kolors/blob/master/imgs/Kolors_paper.pdf).
* 2024.07.03 📊 Kolors won the second place on [FlagEval Multimodal Text-to-Image Leaderboard](https://flageval.baai.ac.cn/#/leaderboard/multimodal?kind=t2i), excelling particularly in the Chinese and English subjective quality assessment where Kolors took the first place.
* 2024.07.02 🎉 Congratulations! Our paper on controllable video generation, [DragAnything: Motion Control for Anything using Entity Representation](https://arxiv.org/abs/2403.07420), have been accepted by ECCV 2024.
* 2024.02.08 🎉 Congratulations! Our paper on generative model evaluation, [Learning Multi-dimensional Human Preference for Text-to-Image Generation](https://wangbohan97.github.io/MPS/), have been accepted by CVPR 2024.
<br><br>

### License Agreement

Kolors are fully open-sourced for academic research. For commercial use, please fill out this [questionnaire](https://github.com/Kwai-Kolors/Kolors/blob/master/imgs/可图KOLORS模型商业授权申请书.docx) and sent it to kwai-kolors@kuaishou.com for registration.

We open-source Kolors to promote the development of large text-to-image models in collaboration with the open-source community. The code of this project is open-sourced under the Apache-2.0 license. We sincerely urge all developers and users to strictly adhere to the [open-source license](MODEL_LICENSE), avoiding the use of the open-source model, code, and its derivatives for any purposes that may harm the country and society or for any services not evaluated and registered for safety. Note that despite our best efforts to ensure the compliance, accuracy, and safety of the data during training, due to the diversity and combinability of generated content and the probabilistic randomness affecting the model, we cannot guarantee the accuracy and safety of the output content, and the model is susceptible to misleading. This project does not assume any legal responsibility for any data security issues, public opinion risks, or risks and liabilities arising from the model being misled, abused, misused, or improperly utilized due to the use of the open-source model and code.


### Reference
[GITHUB](https://github.com/Kwai-Kolors/Kolors/)

[Hugging Face](https://huggingface.co/Kwai-Kolors/)
