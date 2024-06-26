

from zerollama.tasks.chat.interface import ChatModel


class DeepSeekLLM(ChatModel):
    family = "deepseek-llm"
    model_kwargs = {"trust_remote_code": True}
    header = ["name", "size", "quantization", "bits", "subtype"]
    info = [
        # name                                           size       quantization(_, GPTQ, AWQ)     bits   subtype
        ["deepseek-ai/deepseek-llm-7b-chat",             "7b",      "",                            "",    "chat"],
        ["deepseek-ai/deepseek-llm-67b-chat",            "67b",     "",                            "",    "chat"],

        ["deepseek-ai/deepseek-moe-16b-chat",            "MoE16B",  "",                            "",    "chat"],

        ["deepseek-ai/DeepSeek-V2-Chat",                 "MoE236B", "",                            "",    "chat"],
        ["deepseek-ai/DeepSeek-V2-Lite-Chat",            "MoE16B",  "",                            "",    "chat"],

        ["deepseek-ai/deepseek-math-7b-rl",              "7b",      "",                            "",    "math"],
        ["deepseek-ai/deepseek-math-7b-instruct",        "7b",      "",                            "",    "math"],

        ["deepseek-ai/deepseek-coder-1.3b-instruct",     "1.3b",    "",                            "",    "coder"],
        ["deepseek-ai/deepseek-coder-7b-instruct-v1.5",  "7b",      "",                            "",    "coder"],
        ["deepseek-ai/deepseek-coder-6.7b-instruct",     "6.7b",    "",                            "",    "coder"],
        ["deepseek-ai/deepseek-coder-33b-instruct",      "33b",     "",                            "",    "coder"],
    ]


if __name__ == '__main__':
    def transformers_test():
        import torch
        from zerollama.microservices.inference.transformers_green.chat import run_test
        from transformers import BitsAndBytesConfig

        for model_name, kwargs in [("deepseek-ai/deepseek-llm-7b-chat", {}),
                                   ("deepseek-ai/DeepSeek-V2-Lite-Chat", {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}),
                                   ("deepseek-ai/DeepSeek-V2-Lite-Chat", {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)})]:
            print(model_name)
            run_test(model_name, stream=False, **kwargs)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name, kwargs in [("deepseek-ai/deepseek-llm-7b-chat", {}),
                                   ("deepseek-ai/DeepSeek-V2-Lite-Chat", {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}),
                                   ("deepseek-ai/DeepSeek-V2-Lite-Chat", {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)})]:
            print(model_name)
            run_test(model_name, stream=True, **kwargs)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

    transformers_test()
