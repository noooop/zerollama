

from zerollama.inference_backend.hf_transformers.main import HuggingFaceTransformersChat


class DeepSeek(HuggingFaceTransformersChat):
    def __init__(self, model_name, device="cuda", **kwargs):
        HuggingFaceTransformersChat.__init__(self, model_name, info_dict, device, **kwargs)


info_header = ["name", "family", "type", "size", "quantization", "bits"]
info = [
    # name                                   family               type    size      quantization(_, GPTQ, AWQ)     bits

    ["deepseek-ai/deepseek-llm-7b-chat",     "deepseek-llm",      "Chat", "7b",     "",                            ""],


    ["deepseek-ai/deepseek-llm-67b-chat",    "deepseek-llm",      "Chat", "67b",    "",                            ""],

]
info_dict = {x[0]: {k: v for k, v in zip(info_header, x)} for x in info}


if __name__ == '__main__':
    import torch

    def run(model_name, model_class, stream=False):
        print("=" * 80)

        model = model_class(model_name, local_files_only=False)
        model.load()
        print(model.model_info)

        prompt = "给我介绍一下大型语言模型。"

        messages = [
            {"role": "user", "content": prompt}
        ]

        if stream:
            for response in model.stream_chat(messages):
                print(response, end="")
            print()
        else:
            print(model.chat(messages))

    for model_name in ["deepseek-ai/deepseek-llm-7b-chat"]:
        run(model_name, DeepSeek, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in ["deepseek-ai/deepseek-llm-7b-chat"]:
        run(model_name, DeepSeek, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)