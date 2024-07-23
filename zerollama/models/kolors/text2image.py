
from zerollama.tasks.text2image.interface import Text2ImageModel


class Kolors(Text2ImageModel):
    family = "Kolors"
    header = ["name", "use_hf_only"]
    info = [
        ["Kwai-Kolors/Kolors-diffusers", True],
    ]
    inference_backend = "zerollama.models.kolors.backend.text2image:Kolors"


if __name__ == '__main__':

    prompt = "可爱猫猫图"

    for model_name, *_ in Kolors.info:
        model = Kolors.get_model(model_name, local_files_only=False)
        model.load()

        image = model.generate(prompt, options={"height": 1280, "width": 768, "guidance_scale": 5.0, "num_inference_steps": 50}).image

        image.save(f'result-{model_name.replace("/", "-")}.jpg')
