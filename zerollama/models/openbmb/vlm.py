
from zerollama.tasks.vlm.interface import VLMModel


class MiniCPMV(VLMModel):
    family = "MiniCPM-V"
    model_kwargs = {}
    header = ["name"]
    info = [
        # name
        ["openbmb/MiniCPM-V"],
        ["openbmb/MiniCPM-V-2"],
        ["openbmb/MiniCPM-V-2_6"],

        ["openbmb/MiniCPM-Llama3-V-2_5"],
        ["openbmb/MiniCPM-Llama3-V-2_5-int4"],
    ]
    inference_backend = "zerollama.models.openbmb.backend.vlm:MiniCPMV"


if __name__ == '__main__':
    import PIL.Image
    import numpy as np

    import os
    from pathlib import Path

    vlm_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/vlm"

    model_name = "openbmb/MiniCPM-V-2_6"

    model = MiniCPMV.get_model(model_name, local_files_only=False)
    model.load()

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

    vlm_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/dla"

    messages = [
        {
            "role": "user",
            "content": "to markdown."
        }
    ]

    images = [vlm_test_path / "input_sample.png"]

    images = [PIL.Image.open(path).convert("RGB") for path in images]
    images = [np.array(image) for image in images]

    answer = model.chat(messages, images)

    print(answer.content)
