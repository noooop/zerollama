

from zerollama.inference_backend.hf_transformers.main import HuggingFaceTransformersChat


class Qwen1_5(HuggingFaceTransformersChat):
    def __init__(self, model_name, device="cuda", **kwargs):
        HuggingFaceTransformersChat.__init__(self, model_name, info_dict, device, **kwargs)


info_header = ["name", "family", "type", "size", "quantization", "bits"]
info = [
    # name                                   family     type    size      quantization(_, GPTQ, AWQ)     bits
    # original
    ["Qwen/Qwen1.5-0.5B-Chat",               "Qwen1.5", "Chat", "0.5B",   "",                            ""],
    ["Qwen/Qwen1.5-1.8B-Chat",               "Qwen1.5", "Chat", "1.8B",   "",                            ""],
    ["Qwen/Qwen1.5-4B-Chat",                 "Qwen1.5", "Chat", "4B",     "",                            ""],
    ["Qwen/Qwen1.5-7B-Chat",                 "Qwen1.5", "Chat", "7B",     "",                            ""],
    ["Qwen/Qwen1.5-14B-Chat",                "Qwen1.5", "Chat", "14B",    "",                            ""],
    ["Qwen/Qwen1.5-32B-Chat",                "Qwen1.5", "Chat", "32B",    "",                            ""],
    ["Qwen/Qwen1.5-72B-Chat",                "Qwen1.5", "Chat", "72B",    "",                            ""],

    # GPTQ-Int8
    ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",     "Qwen1.5", "Chat", "0.5B",   "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8",     "Qwen1.5", "Chat", "1.8B",   "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-4B-Chat-GPTQ-Int8",       "Qwen1.5", "Chat", "4B",     "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",       "Qwen1.5", "Chat", "7B",     "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",      "Qwen1.5", "Chat", "14B",    "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-32B-Chat-GPTQ-Int8",      "Qwen1.5", "Chat", "32B",    "GPTQ",                        "Int8"],
    ["Qwen/Qwen1.5-72B-Chat-GPTQ-Int8",      "Qwen1.5", "Chat", "72B",    "GPTQ",                        "Int8"],

    # GPTQ-Int4
    ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",     "Qwen1.5", "Chat", "0.5B",   "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4",     "Qwen1.5", "Chat", "1.8B",   "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-4B-Chat-GPTQ-Int4",       "Qwen1.5", "Chat", "4B",     "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",       "Qwen1.5", "Chat", "7B",     "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-14B-Chat-GPTQ-Int4",      "Qwen1.5", "Chat", "14B",    "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",      "Qwen1.5", "Chat", "32B",    "GPTQ",                        "Int4"],
    ["Qwen/Qwen1.5-72B-Chat-GPTQ-Int4",      "Qwen1.5", "Chat", "72B",    "GPTQ",                        "Int4"],

    # AWQ
    ["Qwen/Qwen1.5-0.5B-Chat-AWQ",           "Qwen1.5", "Chat", "0.5B",   "AWQ",                         ""],
    ["Qwen/Qwen1.5-1.8B-Chat-AWQ",           "Qwen1.5", "Chat", "1.8B",   "AWQ",                         ""],
    ["Qwen/Qwen1.5-4B-Chat-AWQ",             "Qwen1.5", "Chat", "4B",     "AWQ",                         ""],
    ["Qwen/Qwen1.5-7B-Chat-AWQ",             "Qwen1.5", "Chat", "7B",     "AWQ",                         ""],
    ["Qwen/Qwen1.5-14B-Chat-AWQ",            "Qwen1.5", "Chat", "14B",    "AWQ",                         ""],
    ["Qwen/Qwen1.5-32B-Chat-AWQ",            "Qwen1.5", "Chat", "32B",    "AWQ",                         ""],
    ["Qwen/Qwen1.5-72B-Chat-AWQ",            "Qwen1.5", "Chat", "72B",    "AWQ",                         ""],
]
info_dict = {x[0]: {k: v for k, v in zip(info_header, x)} for x in info}


if __name__ == '__main__':
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

    for model_name in ["Qwen/Qwen1.5-0.5B-Chat",
                      #"Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",
                       "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",
                       "Qwen/Qwen1.5-0.5B-Chat-AWQ"]:
        run(model_name, Qwen1_5, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in ["Qwen/Qwen1.5-0.5B-Chat",
                      #"Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",
                       "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",
                       "Qwen/Qwen1.5-0.5B-Chat-AWQ"]:
        run(model_name, Qwen1_5, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)