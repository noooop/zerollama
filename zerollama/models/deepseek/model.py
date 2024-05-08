

from zerollama.tasks.chat.interface import ChatModel


class DeepSeekLLM(ChatModel):
    family = "deepseek-llm"
    model_kwargs = {}
    header = ["name", "size", "quantization", "bits"]
    info = [
        # name                                           size      quantization(_, GPTQ, AWQ)     bits   subtype
        ["deepseek-ai/deepseek-vl-1.3b-chat",            "1.3b",   "",                            "",    "chat"],
        ["deepseek-ai/deepseek-vl-7b-chat",              "67b",    "",                            "",    "chat"],

        ["deepseek-ai/deepseek-llm-7b-chat",             "7b",     "",                            "",    "chat"],
        ["deepseek-ai/deepseek-llm-67b-chat",            "67b",    "",                            "",    "chat"],

        ["deepseek-ai/DeepSeek-V2-Chat",                 "MoE21B", "",                            "",    "chat"],
        ["deepseek-ai/deepseek-moe-16b-chat",            "MoE16B", "",                            "",    "chat"],

        ["deepseek-ai/deepseek-math-7b-rl",              "7b",     "",                            "",    "math"],
        ["deepseek-ai/deepseek-math-7b-instruct",        "7b",     "",                            "",    "math"],

        ["deepseek-ai/deepseek-coder-1.3b-instruct",     "1.3b",   "",                            "",    "coder"],
        ["deepseek-ai/deepseek-coder-7b-instruct-v1.5",  "7b",     "",                            "",    "coder"],
        ["deepseek-ai/deepseek-coder-6.7b-instruct",     "6.7b",   "",                            "",    "coder"],
        ["deepseek-ai/deepseek-coder-33b-instruct",      "33b",    "",                            "",    "coder"],
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