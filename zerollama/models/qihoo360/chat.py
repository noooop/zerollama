from zerollama.tasks.chat.interface import ChatModel


class Zhinao360(ChatModel):
    family = "360Zhinao"
    model_kwargs = {"trust_remote_code": True, "use_generation_config": True}
    header = ["name", "size", "quantization", "bits", "subtype"]
    info = [
        # name                                   size      quantization(_, GPTQ, AWQ)     bits,     subtype
        # original
        ["qihoo360/360Zhinao-7B-Chat-4K",        "7B",     "",                            "",       "chat"],
        ["qihoo360/360Zhinao-7B-Chat-32K",       "7B",     "",                            "",       "chat"],
        ["qihoo360/360Zhinao-7B-Chat-360K",      "7B",     "",                            "",       "chat"],


        # GPTQ-Int4
        ["qihoo360/360Zhinao-7B-Chat-4K-Int4",    "7B",   "GPTQ",                        "4bits",  "chat"],
        ["qihoo360/360Zhinao-7B-Chat-32K-Int4",   "7B",   "GPTQ",                        "4bits",  "chat"],
        ["qihoo360/360Zhinao-7B-Chat-360K-Int4",  "7B",   "GPTQ",                        "4bits",  "chat"],
    ]


if __name__ == '__main__':
    import torch

    def transformers_test():
        from zerollama.microservices.inference.transformers_green.chat import run_test
        for model_name, kwargs in [("qihoo360/360Zhinao-7B-Chat-4K-Int4", {})]:
            print(model_name)
            run_test(model_name, stream=False, **kwargs)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name, kwargs in [("qihoo360/360Zhinao-7B-Chat-4K-Int4", {})]:
            print(model_name)
            run_test(model_name, stream=True, **kwargs)
            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)


    transformers_test()