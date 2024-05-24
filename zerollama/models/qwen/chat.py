from zerollama.tasks.chat.interface import ChatModel, ChatGGUFModel


class Qwen1_5(ChatModel):
    family = "Qwen1.5"
    model_kwargs = {}
    header = ["name", "size", "quantization", "bits", "subtype"]
    info = [
        # name                                   size      quantization(_, GPTQ, AWQ)     bits,     subtype
        # original
        ["Qwen/Qwen1.5-0.5B-Chat",                "0.5B",   "",                            "",       "chat"],
        ["Qwen/Qwen1.5-1.8B-Chat",                "1.8B",   "",                            "",       "chat"],
        ["Qwen/Qwen1.5-4B-Chat",                  "4B",     "",                            "",       "chat"],
        ["Qwen/Qwen1.5-7B-Chat",                  "7B",     "",                            "",       "chat"],
        ["Qwen/Qwen1.5-14B-Chat",                 "14B",    "",                            "",       "chat"],
        ["Qwen/Qwen1.5-32B-Chat",                 "32B",    "",                            "",       "chat"],
        ["Qwen/Qwen1.5-72B-Chat",                 "72B",    "",                            "",       "chat"],
        ["Qwen/Qwen1.5-110B-Chat",                "110B",   "",                            "",       "chat"],

        # GPTQ-Int4
        ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",      "0.5B",   "GPTQ",                        "4bits",  "chat"],
        ["Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4",      "1.8B",   "GPTQ",                        "4bits",  "chat"],
        ["Qwen/Qwen1.5-4B-Chat-GPTQ-Int4",        "4B",     "GPTQ",                        "4bits",  "chat"],
        ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",        "7B",     "GPTQ",                        "4bits",  "chat"],
        ["Qwen/Qwen1.5-14B-Chat-GPTQ-Int4",       "14B",    "GPTQ",                        "4bits",  "chat"],
        ["Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",       "32B",    "GPTQ",                        "4bits",  "chat"],
        ["Qwen/Qwen1.5-72B-Chat-GPTQ-Int4",       "72B",    "GPTQ",                        "4bits",  "chat"],
        ["Qwen/Qwen1.5-110B-Chat-GPTQ-Int4",      "110B",   "GPTQ",                        "4bits",  "chat"],

        # AWQ
        ["Qwen/Qwen1.5-0.5B-Chat-AWQ",            "0.5B",   "AWQ",                         "4bits",  "chat"],
        ["Qwen/Qwen1.5-1.8B-Chat-AWQ",            "1.8B",   "AWQ",                         "4bits",  "chat"],
        ["Qwen/Qwen1.5-4B-Chat-AWQ",              "4B",     "AWQ",                         "4bits",  "chat"],
        ["Qwen/Qwen1.5-7B-Chat-AWQ",              "7B",     "AWQ",                         "4bits",  "chat"],
        ["Qwen/Qwen1.5-14B-Chat-AWQ",             "14B",    "AWQ",                         "4bits",  "chat"],
        ["Qwen/Qwen1.5-32B-Chat-AWQ",             "32B",    "AWQ",                         "4bits",  "chat"],
        ["Qwen/Qwen1.5-72B-Chat-AWQ",             "72B",    "AWQ",                         "4bits",  "chat"],
        ["Qwen/Qwen1.5-110B-Chat-AWQ",            "110B",   "AWQ",                         "4bits",  "chat"],

        # MoE
        ["Qwen/Qwen1.5-MoE-A2.7B-Chat",           "A2.7B",  "",                            "",       "chat"],
        ["Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4", "A2.7B",  "GPTQ",                        "4bits",  "chat"],

        # CodeQwen1.5
        ["Qwen/CodeQwen1.5-7B-Chat",              "7B",     "",                            "",       "coder"],
        ["Qwen/CodeQwen1.5-7B-Chat-AWQ",          "7B",     "AWQ",                         "4bits",  "coder"],
    ]


class Qwen1_5_GGUF(ChatGGUFModel):
    family = "Qwen1.5_gguf"

    gguf = {
        "repo_id": [
            "Qwen/Qwen1.5-0.5B-Chat-GGUF",
            "Qwen/Qwen1.5-1.8B-Chat-GGUF",
            "Qwen/Qwen1.5-4B-Chat-GGUF",
            "Qwen/Qwen1.5-7B-Chat-GGUF",
            "Qwen/Qwen1.5-14B-Chat-GGUF",
            "Qwen/Qwen1.5-32B-Chat-GGUF",
            "Qwen/Qwen1.5-72B-Chat-GGUF",
            "Qwen/Qwen1.5-110B-Chat-GGUF",
        ],
        "filename": [
            "*q8_0.gguf",
            "*q6_k.gguf",
            "*q5_k_m.gguf",
            "*q5_0.gguf",
            "*q4_k_m.gguf",
            "*q4_0.gguf",
            "*q3_k_m.gguf",
            "*q2_k.gguf"
        ]
    }


if __name__ == '__main__':
    import torch

    def transformers_test():
        from zerollama.microservices.inference.transformers_green.chat import run_test
        from transformers import BitsAndBytesConfig

        for model_name, kwargs in [("Qwen/Qwen1.5-0.5B-Chat", {}),
                                   ("Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4", {}),
                                   ("Qwen/Qwen1.5-0.5B-Chat-AWQ", {}),
                                   ("Qwen/Qwen1.5-MoE-A2.7B-Chat", {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}),
                                   ("Qwen/Qwen1.5-MoE-A2.7B-Chat", {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)})]:
            print(model_name)
            run_test(model_name, stream=False, **kwargs)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name, kwargs in [("Qwen/Qwen1.5-0.5B-Chat", {}),
                                   ("Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4", {}),
                                   ("Qwen/Qwen1.5-0.5B-Chat-AWQ", {}),
                                   ("Qwen/Qwen1.5-MoE-A2.7B-Chat", {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}),
                                   ("Qwen/Qwen1.5-MoE-A2.7B-Chat", {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)})]:
            print(model_name)
            run_test(model_name, stream=True, **kwargs)
            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

    def llama_cpp_test():
        from zerollama.microservices.inference.llama_cpp_green.chat import run_test

        for model_name in ["Qwen/Qwen1.5-0.5B-Chat-GGUF+*q8_0.gguf",
                           "Qwen/Qwen1.5-0.5B-Chat-GGUF+*q2_k.gguf"]:
            print(model_name)
            run_test(model_name, stream=False)

        for model_name in ["Qwen/Qwen1.5-0.5B-Chat-GGUF+*q8_0.gguf",
                           "Qwen/Qwen1.5-0.5B-Chat-GGUF+*q2_k.gguf"]:
            print(model_name)
            run_test(model_name, stream=True)

    #transformers_test()
    llama_cpp_test()