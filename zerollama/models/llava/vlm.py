from zerollama.tasks.vlm.interface import VLMModel


class Llava(VLMModel):
    family = "Llava"
    header = ["name", "use_hf_only"]
    info = [
        ["llava-hf/llava-v1.6-vicuna-7b-hf",         True],
        ["llava-hf/llava-v1.6-vicuna-13b-hf",        True],
        ["llava-hf/llava-1.5-7b-hf",                 True],
        ["llava-hf/llava-1.5-13b-hf",                True],
        ["llava-hf/bakLlava-v1-hf",                  True],
        ["llava-hf/llava-v1.6-mistral-7b-hf",        True],
        ["llava-hf/llava-v1.6-34b-hf",               True],
        ["llava-hf/vip-llava-13b-hf",                True],
        ["llava-hf/vip-llava-7b-hf",                 True],
        ["llava-hf/llava-interleave-qwen-7b-dpo-hf", True],
        ["llava-hf/llava-interleave-qwen-7b-hf",     True],
        ["llava-hf/llava-interleave-qwen-0.5b-hf",   True],
    ]
    inference_backend = "zerollama.models.llava.backend.vlm:Llava"


if __name__ == '__main__':
    import PIL.Image
    import os
    from pathlib import Path

    model_name = Llava.info[0][0]
    model = Llava.get_model(model_name, local_files_only=False)
    model.load()

    messages = [
        {
            "role": "user",
            "content": "这个图片的内容是什么。"
        }
    ]

    vlm_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/vlm"

    images = [vlm_test_path / "monday.jpg"]

    images = [PIL.Image.open(path).convert("RGB") for path in images]

    response = model.chat(messages, images)
    print(response.content)