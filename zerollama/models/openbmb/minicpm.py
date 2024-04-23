

from zerollama.inference_backend.hf_transformers.main import HuggingFaceTransformersChat


class MiniCPM(HuggingFaceTransformersChat):
    def __init__(self, model_name, device="cuda", **kwargs):
        super().__init__(model_name, info_dict, device, trust_remote_code=True, **kwargs)


info_header = ["name", "family", "type", "size", "quantization", "bits", "torch_dtype"]
info = [
    # name                                        family     type    size    quantization(_, GPTQ, AWQ)     bits, torch_dtype
    ["openbmb/MiniCPM-2B-sft-fp32",               "MiniCPM", "Chat", "2B",   "",                            "",   "fp32"],
    ["openbmb/MiniCPM-2B-sft-bf16",               "MiniCPM", "Chat", "2B",   "",                            "",   "bf16"],
    ["openbmb/MiniCPM-2B-sft-int4",               "MiniCPM", "Chat", "2B",   "",                            "",   "int4"],

    ["openbmb/MiniCPM-2B-sft-fp32-llama-format",  "MiniCPM", "Chat", "2B",   "",                            "",   "fp32"],
    ["openbmb/MiniCPM-2B-sft-bf16-llama-format",  "MiniCPM", "Chat", "2B",   "",                            "",   "bf16"],
    ["openbmb/MiniCPM-2B-dpo-bf16-llama-format",  "MiniCPM", "Chat", "2B",   "",                            "",   "bf16"],

    ["openbmb/MiniCPM-2B-dpo-fp32",               "MiniCPM", "Chat", "2B",   "",                            "",   "fp32"],
    ["openbmb/MiniCPM-2B-dpo-fp16",               "MiniCPM", "Chat", "2B",   "",                            "",   "fp16"],
    ["openbmb/MiniCPM-2B-dpo-bf16",               "MiniCPM", "Chat", "2B",   "",                            "",   "bf16"],
    ["openbmb/MiniCPM-2B-dpo-int4",               "MiniCPM", "Chat", "2B",   "",                            "",   "int4"],

    ["openbmb/MiniCPM-1B-sft-bf16",               "MiniCPM", "Chat", "1B",   "",                            "",   "bf16"],
    ["openbmb/MiniCPM-2B-128k",                   "MiniCPM", "Chat", "2B",   "",                            "",   "fp32"],
    ["openbmb/MiniCPM-MoE-8x2B",                  "MiniCPM", "Chat", "8x2B", "",                            "",   "fp32"],
]
info_dict = {x[0]: {k: v for k, v in zip(info_header, x)} for x in info}


if __name__ == '__main__':
    import gc
    import torch

    def run(model_name, model_class, stream=False):
        print("=" * 80)

        model = model_class(model_name, local_files_only=False)
        model.load()
        print(model.model_info)

        prompt = "给我介绍一下大型语言模型。"

        messages = [
            {"role": "user", "content": prompt}
        ]

        if stream:
            for response in model.stream_chat(messages):
                print(response, end="")
            print()
        else:
            print(model.chat(messages))

    for model_name in ["openbmb/MiniCPM-2B-sft-bf16",
                       "openbmb/MiniCPM-2B-dpo-bf16"]:
        run(model_name, MiniCPM, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in ["openbmb/MiniCPM-2B-sft-bf16",
                       "openbmb/MiniCPM-2B-dpo-bf16"]:
        run(model_name, MiniCPM, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)