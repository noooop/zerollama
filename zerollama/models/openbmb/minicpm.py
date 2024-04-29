
from zerollama.tasks.chat.interface import ChatModel


class MiniCPM(ChatModel):
    family = "MiniCPM"
    model_kwargs = {"trust_remote_code": True}
    header = ["name", "size", "quantization", "bits", "torch_dtype"]
    info = [
        # name                                          size    quantization(_, GPTQ, AWQ)     bits, torch_dtype
        ["openbmb/MiniCPM-2B-sft-fp32",                 "2B",   "",                            "",   "fp32"],
        ["openbmb/MiniCPM-2B-sft-bf16",                 "2B",   "",                            "",   "bf16"],
        ["openbmb/MiniCPM-2B-sft-int4",                 "2B",   "",                            "",   "int4"],

        ["openbmb/MiniCPM-2B-sft-fp32-llama-format",    "2B",   "",                            "",   "fp32"],
        ["openbmb/MiniCPM-2B-sft-bf16-llama-format",    "2B",   "",                            "",   "bf16"],
        ["openbmb/MiniCPM-2B-dpo-bf16-llama-format",    "2B",   "",                            "",   "bf16"],

        ["openbmb/MiniCPM-2B-dpo-fp32",                 "2B",   "",                            "",   "fp32"],
        ["openbmb/MiniCPM-2B-dpo-fp16",                 "2B",   "",                            "",   "fp16"],
        ["openbmb/MiniCPM-2B-dpo-bf16",                 "2B",   "",                            "",   "bf16"],
        ["openbmb/MiniCPM-2B-dpo-int4",                 "2B",   "",                            "",   "int4"],

        ["openbmb/MiniCPM-1B-sft-bf16",                 "1B",   "",                            "",   "bf16"],
        ["openbmb/MiniCPM-2B-128k",                     "2B",   "",                            "",   "fp32"],
        ["openbmb/MiniCPM-MoE-8x2B",                    "8x2B", "",                            "",   "fp32"],
    ]


if __name__ == '__main__':
    import torch
    from zerollama.inference_backend.hf_transformers.chat import run_test

    for model_name in ["openbmb/MiniCPM-2B-sft-bf16",
                       "openbmb/MiniCPM-2B-dpo-bf16"]:
        run_test(model_name, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in ["openbmb/MiniCPM-2B-sft-bf16",
                       "openbmb/MiniCPM-2B-dpo-bf16"]:
        run_test(model_name, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)