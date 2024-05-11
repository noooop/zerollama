from zerollama.tasks.chat.interface import ChatModel, ChatGGUFModel


class Qwen1_5(ChatModel):
    family = "Qwen1.5"
    model_kwargs = {}
    header = ["name", "size", "quantization", "bits"]
    info = [
        # name                                   size      quantization(_, GPTQ, AWQ)     bits
        # original
        ["Qwen/Qwen1.5-0.5B-Chat",               "0.5B",   "",                            ""],
        ["Qwen/Qwen1.5-1.8B-Chat",               "1.8B",   "",                            ""],
        ["Qwen/Qwen1.5-4B-Chat",                 "4B",     "",                            ""],
        ["Qwen/Qwen1.5-7B-Chat",                 "7B",     "",                            ""],
        ["Qwen/Qwen1.5-14B-Chat",                "14B",    "",                            ""],
        ["Qwen/Qwen1.5-32B-Chat",                "32B",    "",                            ""],
        ["Qwen/Qwen1.5-72B-Chat",                "72B",    "",                            ""],
        ["Qwen/Qwen1.5-110B-Chat",               "110B",   "",                            ""],

        # GPTQ-Int8
        # ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",     "0.5B",   "GPTQ",                        "8bits"],
        # ["Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8",     "1.8B",   "GPTQ",                        "8bits"],
        # ["Qwen/Qwen1.5-4B-Chat-GPTQ-Int8",       "4B",     "GPTQ",                        "8bits"],
        # ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",       "7B",     "GPTQ",                        "8bits"],
        # ["Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",      "14B",    "GPTQ",                        "8bits"],
        # ["Qwen/Qwen1.5-32B-Chat-GPTQ-Int8",      "32B",    "GPTQ",                        "8bits"],
        # ["Qwen/Qwen1.5-72B-Chat-GPTQ-Int8",      "72B",    "GPTQ",                        "8bits"],

        # GPTQ-Int4
        ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",     "0.5B",   "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4",     "1.8B",   "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-4B-Chat-GPTQ-Int4",       "4B",     "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",       "7B",     "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-14B-Chat-GPTQ-Int4",      "14B",    "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",      "32B",    "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-72B-Chat-GPTQ-Int4",      "72B",    "GPTQ",                        "4bits"],
        ["Qwen/Qwen1.5-110B-Chat-GPTQ-Int4",     "110B",   "GPTQ",                        "4bits"],

        # AWQ
        ["Qwen/Qwen1.5-0.5B-Chat-AWQ",           "0.5B",   "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-1.8B-Chat-AWQ",           "1.8B",   "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-4B-Chat-AWQ",             "4B",     "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-7B-Chat-AWQ",             "7B",     "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-14B-Chat-AWQ",            "14B",    "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-32B-Chat-AWQ",            "32B",    "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-72B-Chat-AWQ",            "72B",    "AWQ",                         "4bits"],
        ["Qwen/Qwen1.5-110B-Chat-AWQ",           "110B",   "AWQ",                         "4bits"],
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
        from zerollama.inference_backend.transformers_green.chat import run_test

        for model_name in ["Qwen/Qwen1.5-0.5B-Chat",
                           "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",
                           "Qwen/Qwen1.5-0.5B-Chat-AWQ"]:
            run_test(model_name, stream=False)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name in ["Qwen/Qwen1.5-0.5B-Chat",
                           "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",
                           "Qwen/Qwen1.5-0.5B-Chat-AWQ"]:
            run_test(model_name, stream=True)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

    def llama_cpp_test():
        from zerollama.inference_backend.llama_cpp_green.chat import run_test

        for model_name in ["Qwen/Qwen1.5-0.5B-Chat-GGUF+*q8_0.gguf",
                           "Qwen/Qwen1.5-0.5B-Chat-GGUF+*q2_k.gguf"]:
            run_test(model_name, stream=False)

        for model_name in ["Qwen/Qwen1.5-0.5B-Chat-GGUF+*q8_0.gguf",
                           "Qwen/Qwen1.5-0.5B-Chat-GGUF+*q2_k.gguf"]:
            run_test(model_name, stream=True)


    #transformers_test()
    llama_cpp_test()