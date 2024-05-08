

from zerollama.tasks.chat.interface import ChatModel


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


if __name__ == '__main__':
    import torch
    from zerollama.inference_backend.transformers_green.chat import run_test

    for model_name in ["xverse/XVERSE-7B-Chat"]:
        run_test(model_name, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in ["xverse/XVERSE-7B-Chat"]:
        run_test(model_name, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)