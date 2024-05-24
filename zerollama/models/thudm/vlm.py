
from zerollama.tasks.vlm.interface import VLMModel


class CogVLM2(VLMModel):
    family = "CogVLM2"
    model_kwargs = {"trust_remote_code": True}
    header = ["name", "hf_name", "modelscope_name"]
    info = [
        # name                                 hf_name                                   modelscope_name
        ["cogvlm2-llama3-chat-19B",            "THUDM/cogvlm2-llama3-chat-19B",          "ZhipuAI/cogvlm2-llama3-chat-19B"],
        ["cogvlm2-llama3-chinese-chat-19B",    "THUDM/cogvlm2-llama3-chinese-chat-19B",  "ZhipuAI/cogvlm2-llama3-chinese-chat-19B"],
    ]
    inference_backend = "zerollama.models.thudm.backend.vlm:CogVLM"


if __name__ == '__main__':
    import PIL.Image
    import numpy as np

    import os
    from pathlib import Path

    vlm_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/vlm_test"

    from transformers import BitsAndBytesConfig

    def get_model(model_name):
        model_kwargs = {"local_files_only": False, "quantization_config": BitsAndBytesConfig(load_in_4bit=True)}

        model_class = CogVLM2.inference_backend
        module_name, class_name = model_class.split(":")
        import importlib

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = model_class(model_name=model_name, **model_kwargs)
        model.load()
        return model


    model_name = "cogvlm2-llama3-chinese-chat-19B"

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

