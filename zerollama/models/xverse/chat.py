

from zerollama.tasks.chat.interface import ChatModel, ChatGGUFModel


class XVERSE(ChatModel):
    family = "xverse"
    model_kwargs = {"trust_remote_code": True}
    header = ["name", "size", "quantization", "bits"]
    info = [
        # name                                 size       quantization(_, GPTQ, AWQ)     bits
        ["xverse/XVERSE-7B-Chat",              "7B",      "",                            ""],
        ["xverse/XVERSE-13B-Chat",             "13B",     "",                            ""],
        ["xverse/XVERSE-65B-Chat",             "65B",     "",                            ""],
        ["xverse/XVERSE-MoE-A4.2B-Chat",       "A4.2B",   "",                            ""],

        ["xverse/XVERSE-7B-Chat-GPTQ-Int8",    "7B",      "GPTQ",                        "8bits"],
        ["xverse/XVERSE-13B-Chat-GPTQ-Int8",   "13B",     "GPTQ",                        "8bits"],
        ["xverse/XVERSE-65B-Chat-GPTQ-Int8",   "65B",     "GPTQ",                        "8bits"],

        ["xverse/XVERSE-7B-Chat-GPTQ-Int4",    "7B",      "GPTQ",                        "4bits"],
        ["xverse/XVERSE-13B-Chat-GPTQ-Int4",   "13B",     "GPTQ",                        "4bits"],
        ["xverse/XVERSE-65B-Chat-GPTQ-Int4",   "65B",     "GPTQ",                        "4bits"],
    ]


class XVERSE_GGUF(ChatGGUFModel):
    family = "xverse_gguf"

    gguf = {
        "repo_id": [
            "xverse/XVERSE-7B-Chat-GGUF",
            "xverse/XVERSE-13B-Chat-GGUF",
            "xverse/XVERSE-65B-Chat-GGUF",
        ],
        "filename": [
            "*fp16.gguf"
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
    from zerollama.inference_backend.transformers_green.chat import run_test


    def transformers_test():
        for model_name in ["xverse/XVERSE-7B-Chat"]:
            run_test(model_name, stream=False)

            print(torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name in ["xverse/XVERSE-7B-Chat"]:
            run_test(model_name, stream=True)

            print(torch.cuda.memory_allocated() / 1024 ** 2)

    def llama_cpp_test():
        from zerollama.inference_backend.llama_cpp_green.chat import run_test

        for model_name in ["xverse/XVERSE-7B-Chat-GGUF+*q4_0.gguf"]:
            run_test(model_name, stream=False)

        for model_name in ["xverse/XVERSE-7B-Chat-GGUF+*q4_0.gguf"]:
            run_test(model_name, stream=True)


    llama_cpp_test()