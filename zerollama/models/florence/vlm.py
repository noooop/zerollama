
from zerollama.tasks.vlm.interface import VLMModel


class Florence(VLMModel):
    family = "Florence"
    model_kwargs = {}
    header = ["name", "use_hf_only"]
    info = [
        # name
        ["microsoft/Florence-2-base-ft",  True],
        ["microsoft/Florence-2-large-ft", True],
    ]
    inference_backend = "zerollama.models.florence.backend.vlm:Florence"


if __name__ == '__main__':
    import PIL.Image
    import numpy as np

    import os
    from pathlib import Path

    vlm_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/vlm"

    for model_name, *_ in Florence.info:
        print(model_name)

        model = Florence.get_model(model_name, local_files_only=False)
        model.load()

        for task in ["<OD>", "<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<OCR>"]:
            messages = [
                {
                    "role": "user",
                    "task": task
                }
            ]

            images = [vlm_test_path / "monday.jpg"]

            images = [PIL.Image.open(path).convert("RGB") for path in images]
            images = [np.array(image) for image in images]

            answer = model.chat(messages, images)

            print(answer.data)

