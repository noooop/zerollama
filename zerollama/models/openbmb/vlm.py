
from zerollama.tasks.vlm.interface import VLMModel


class MiniCPMV(VLMModel):
    family = "MiniCPM-V"
    model_kwargs = {}
    header = ["name"]
    info = [
        # name
        ["openbmb/MiniCPM-V"],
        ["openbmb/MiniCPM-V-2"],

        ["openbmb/MiniCPM-Llama3-V-2_5"],
        ["openbmb/MiniCPM-Llama3-V-2_5-int4"],
    ]
    inference_backend = "zerollama.models.openbmb.backend.vlm:MiniCPMV"


if __name__ == '__main__':
    import PIL.Image
    import numpy as np

    import os
    from pathlib import Path

    vlm_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/vlm_test"

    def get_model(model_name):
        model_kwargs = {}

        model_class = MiniCPMV.inference_backend
        module_name, class_name = model_class.split(":")
        import importlib

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = model_class(model_name=model_name, **model_kwargs)
        model.load()
        return model


    model_name = "openbmb/MiniCPM-V"

    model = get_model(model_name)

    messages = [
        {
            "role": "user",
            "content": "这个图片的内容是什么。"
        }
    ]

    images = [vlm_test_path / "monday.jpg"]

    images = [PIL.Image.open(path).convert("RGB") for path in images]
    images = [np.array(image) for image in images]

    answer = model.chat(messages, images)

    print(answer.content)

