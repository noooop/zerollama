

from zerollama.tasks.chat.interface import ChatModel


class DeepSeekLLM(ChatModel):
    family = "deepseek-llm"
    model_kwargs = {}
    header = ["name", "size", "quantization", "bits"]
    info = [
        # name                                 size      quantization(_, GPTQ, AWQ)     bits
        ["deepseek-ai/deepseek-llm-7b-chat",   "7b",     "",                            ""],

        ["deepseek-ai/deepseek-llm-67b-chat",  "67b",    "",                            ""],

    ]


if __name__ == '__main__':
    import torch
    from zerollama.inference_backend.transformers_green.chat import run_test

    for model_name in ["deepseek-ai/deepseek-llm-7b-chat"]:
        run_test(model_name, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in ["deepseek-ai/deepseek-llm-7b-chat"]:
        run_test(model_name, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)