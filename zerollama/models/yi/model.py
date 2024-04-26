

from zerollama.core.models.chat import ChatModel


class Yi(ChatModel):
    family = "Yi"
    model_kwargs = {}
    header = ["name", "size", "quantization", "bits"]
    info = [
        # name                         size    quantization(_, GPTQ, AWQ)     bits
        ["01-ai/Yi-6B-Chat",           "6B",   "",                            ""],
        ["01-ai/Yi-6B-Chat-8bits",     "6B",   "GPTQ",                        "8bits"],
        ["01-ai/Yi-6B-Chat-4bits",     "6B",   "AWQ",                         "4bits"],

        ["01-ai/Yi-34B-Chat",          "34B",  "",                            ""],
        ["01-ai/Yi-34B-Chat-8bits",    "34B",  "GPTQ",                        "8bits"],
        ["01-ai/Yi-34B-Chat-4bits",    "34B",  "AWQ",                         "4bits"],
    ]


if __name__ == '__main__':
    import torch
    from zerollama.inference_backend.hf_transformers.main import run_test

    for model_name in [#"01-ai/Yi-6B-Chat",
                       #"01-ai/Yi-6B-Chat-8bits",
                       "01-ai/Yi-6B-Chat-4bits"]:
        run_test(model_name, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in [#"01-ai/Yi-6B-Chat",
                       #"01-ai/Yi-6B-Chat-8bits",
                       "01-ai/Yi-6B-Chat-4bits"]:
        run_test(model_name, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)