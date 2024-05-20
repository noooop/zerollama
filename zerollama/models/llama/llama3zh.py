
from zerollama.tasks.chat.interface import ChatModel


class Llama3ZH(ChatModel):
    family = "llama3zh"
    model_kwargs = {}
    header = ["name", "size", "quantization", "bits"]
    info = [
        # name                                           size      quantization(_, GPTQ, AWQ)     bits
        ["UnicomLLM/Unichat-llama3-Chinese-8B",          "8B",     "",                            ""],
        ["hfl/llama-3-chinese-8b-instruct",              "8B",     "",                            ""],
    ]


if __name__ == '__main__':
    import torch
    from zerollama.microservices.inference.transformers_green.chat import run_test

    for model_name in ["hfl/llama-3-chinese-8b-instruct"]:
        run_test(model_name, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in ["hfl/llama-3-chinese-8b-instruct"]:
        run_test(model_name, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)