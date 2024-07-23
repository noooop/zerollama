
from zerollama.tasks.text2image.interface import Text2ImageModel


class HunyuanDiT(Text2ImageModel):
    family = "HunyuanDiT"
    header = ["name", "use_hf_only"]
    info = [
        ["Tencent-Hunyuan/HunyuanDiT-Diffusers", True],
        ["Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled", True],

        ["Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers", True],
        ["Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled", True],

        ["Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", True],
        ["Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled", True],
    ]
    inference_backend = "zerollama.models.hunyun.backend.text2image:HunyuanDiT"


if __name__ == '__main__':

    prompt = "可爱猫猫图"

    for model_name, *_ in HunyuanDiT.info:
        model = HunyuanDiT.get_model(model_name, local_files_only=False)
        model.load()

        image = model.generate(prompt, options={"height": 1280, "width": 768, "num_inference_steps": 50}).image

        image.save(f'result-{model_name.replace("/", "-")}.jpg')
