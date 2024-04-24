

from zerollama.inference_backend.hf_transformers.main import HuggingFaceTransformersChat


class Yi(HuggingFaceTransformersChat):
    def __init__(self, model_name, device="cuda", **kwargs):
        HuggingFaceTransformersChat.__init__(self, model_name, info_dict, device, **kwargs)


info_header = ["name", "family", "type", "size", "quantization", "bits"]
info = [
    # name                                   family     type    size      quantization(_, GPTQ, AWQ)     bits

    ["01-ai/Yi-6B-Chat",                     "Yi",      "Chat", "6B",     "",                            ""],
    ["01-ai/Yi-6B-Chat-8bits",               "Yi",      "Chat", "6B",     "GPTQ",                        "8bits"],
    ["01-ai/Yi-6B-Chat-4bits",               "Yi",      "Chat", "6B",     "AWQ",                         "4bits"],

    ["01-ai/Yi-34B-Chat",                    "Yi",      "Chat", "34B",    "",                            ""],
    ["01-ai/Yi-34B-Chat-8bits",              "Yi",      "Chat", "34B",    "GPTQ",                        "8bits"],
    ["01-ai/Yi-34B-Chat-4bits",              "Yi",      "Chat", "34B",    "AWQ",                         "4bits"],
]
info_dict = {x[0]: {k: v for k, v in zip(info_header, x)} for x in info}


if __name__ == '__main__':
    import torch
    from zerollama.inference_backend.hf_transformers.main import run_test

    for model_name in [#"01-ai/Yi-6B-Chat",
                       #"01-ai/Yi-6B-Chat-8bits",
                       "01-ai/Yi-6B-Chat-4bits"]:
        run_test(model_name, Yi, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in [#"01-ai/Yi-6B-Chat",
                       #"01-ai/Yi-6B-Chat-8bits",
                       "01-ai/Yi-6B-Chat-4bits"]:
        run_test(model_name, Yi, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)