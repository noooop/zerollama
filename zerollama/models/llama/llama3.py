
from zerollama.tasks.chat.interface import ChatModel


class Llama3(ChatModel):
    family = "llama3"
    model_kwargs = {}
    header = ["name", "modelscope_name", "size", "quantization", "bits"]
    info = [
        # name                                          modelscope_name                             size    quantization(_, GPTQ, AWQ)     bits
        ["meta-llama/Meta-Llama-3-8B-Instruct",         "LLM-Research/Meta-Llama-3-8B-Instruct",     "8B",   "",                            ""],
        ["meta-llama/Meta-Llama-3-70B-Instruct",        "LLM-Research/Meta-Llama-3-70B-Instruct",    "70B",  "",                            ""],
    ]


class Llama3_1(ChatModel):
    family = "llama3.1"
    model_kwargs = {}
    header = ["name", "modelscope_name", "size", "quantization", "bits"]
    info = [
        # name                                                modelscope_name                                      size    quantization(_, GPTQ, AWQ)     bits
        ["meta-llama/Meta-Llama-3.1-8B-Instruct",            "LLM-Research/Meta-Llama-3.1-8B-Instruct",            "8B",   "",                            ""],
        ["meta-llama/Meta-Llama-3.1-70B-Instruct",           "LLM-Research/Meta-Llama-3.1-70B-Instruct",           "70B",  "",                            ""],
        ["meta-llama/Meta-Llama-3.1-405B-Instruct",          "LLM-Research/Meta-Llama-3.1-405B-Instruct",          "405B", "",                            ""],

        ["meta-llama/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",   "LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",   "8B",   "AWQ",                         "4bit"],
        ["meta-llama/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",  "LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",  "70B",  "AWQ",                         "4bit"],
        ["meta-llama/Meta-Llama-3.1-405B-Instruct-AWQ-INT4", "LLM-Research/Meta-Llama-3.1-405B-Instruct-AWQ-INT4", "405B", "AWQ",                         "4bit"],
    ]


if __name__ == '__main__':
    import torch
    from zerollama.microservices.inference.transformers_green.chat import run_test

    for model_name in ["meta-llama/Meta-Llama-3.1-8B-Instruct"]:
        run_test(model_name, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in ["meta-llama/Meta-Llama-3.1-8B-Instruct"]:
        run_test(model_name, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)