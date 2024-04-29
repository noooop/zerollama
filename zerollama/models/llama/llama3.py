
from zerollama.tasks.chat.interface import ChatModel


class Llama3(ChatModel):
    family = "llama3"
    model_kwargs = {}
    header = ["name", "size", "quantization", "bits"]
    info = [
        # name                                          size    quantization(_, GPTQ, AWQ)     bits
        ["meta-llama/Meta-Llama-3-8B-Instruct",         "8B",   "",                            ""],
        ["meta-llama/Meta-Llama-3-70B-Instruct",        "70B",  "",                            ""],
    ]


if __name__ == '__main__':
    import torch
    from zerollama.inference_backend.hf_transformers.chat import run_test

    for model_name in ["meta-llama/Meta-Llama-3-8B-Instruct"]:
        run_test(model_name, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in ["meta-llama/Meta-Llama-3-8B-Instruct"]:
        run_test(model_name, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)