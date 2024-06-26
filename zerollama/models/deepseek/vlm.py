
from zerollama.tasks.vlm.interface import VLMModel


class DeepSeekVL(VLMModel):
    family = "DeepSeek-VL"
    model_kwargs = {}
    header = ["name", "size"]
    info = [
        # name                                           size
        ["deepseek-ai/deepseek-vl-1.3b-chat",            "1.3b"],
        ["deepseek-ai/deepseek-vl-7b-chat",              "67b"],
    ]
    inference_backend = "zerollama.models.deepseek.backend.vlm:DeepseekVL"


if __name__ == '__main__':
    import PIL.Image
    import numpy as np

    import os
    from pathlib import Path

    vlm_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/vlm"

    def get_model(model_name):
        model_kwargs = {}

        model_class = DeepSeekVL.inference_backend
        module_name, class_name = model_class.split(":")
        import importlib

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = model_class(model_name=model_name, **model_kwargs)
        model.load()
        return model


    model_name = "deepseek-ai/deepseek-vl-1.3b-chat"

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

    response = model.chat(messages, images)
    print(response.content)

