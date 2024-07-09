
from zerollama.tasks.chat.interface import ChatModel, ChatGGUFModel


class MiniCPM(ChatModel):
    family = "MiniCPM"
    model_kwargs = {"trust_remote_code": True}
    header = ["name", "modelscope_name", "size", "quantization", "bits", "torch_dtype"]
    info = [
        # name                                          modelscope_name                size    quantization(_, GPTQ, AWQ)     bits, torch_dtype
        ["openbmb/MiniCPM-2B-sft-fp32",                 "",                            "2B",   "",                            "",   "fp32"],
        ["openbmb/MiniCPM-2B-sft-bf16",                 "OpenBMB/miniCPM-bf16",        "2B",   "",                            "",   "bf16"],
        ["openbmb/MiniCPM-2B-sft-int4",                 "",                            "2B",   "",                            "",   "int4"],

        ["openbmb/MiniCPM-2B-sft-fp32-llama-format",    "",                            "2B",   "",                            "",   "fp32"],
        ["openbmb/MiniCPM-2B-sft-bf16-llama-format",    "",                            "2B",   "",                            "",   "bf16"],
        ["openbmb/MiniCPM-2B-dpo-bf16-llama-format",    "",                            "2B",   "",                            "",   "bf16"],

        ["openbmb/MiniCPM-2B-dpo-fp32",                 "",                            "2B",   "",                            "",   "fp32"],
        ["openbmb/MiniCPM-2B-dpo-fp16",                 "",                            "2B",   "",                            "",   "fp16"],
        ["openbmb/MiniCPM-2B-dpo-bf16",                 "OpenBMB/MiniCPM-2B-dpo-bf16", "2B",   "",                            "",   "bf16"],
        ["openbmb/MiniCPM-2B-dpo-int4",                 "",                            "2B",   "",                            "",   "int4"],

        ["openbmb/MiniCPM-1B-sft-bf16",                 "MiniCPM-1B-sft-bf16",         "1B",   "",                            "",   "bf16"],
        ["openbmb/MiniCPM-2B-128k",                     "openbmb/MiniCPM-2B-128k",     "2B",   "",                            "",   "fp32"],
        ["openbmb/MiniCPM-MoE-8x2B",                    "OpenBMB/MiniCPM-MoE-8x2B",    "8x2B", "",                            "",   "fp32"],

        ["openbmb/MiniCPM-S-1B-sft",                    "openbmb/MiniCPM-S-1B-sft",    "1B",   "",                            "",   "bf16"],
        ["openbmb/MiniCPM-S-1B-sft-llama-format",       "",                            "1B",   "",                            "",   "bf16"],
    ]


if __name__ == '__main__':
    def transformers_test():
        import torch
        from zerollama.microservices.inference.transformers_green.chat import run_test

        for model_name in ["openbmb/MiniCPM-S-1B-sft"]:
            print(model_name)
            run_test(model_name, stream=False)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name in ["openbmb/MiniCPM-S-1B-sft"]:
            print(model_name)
            run_test(model_name, stream=True)
            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

    transformers_test()
