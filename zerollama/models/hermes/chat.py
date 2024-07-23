from zerollama.tasks.chat.interface import ChatModel


class Hermes(ChatModel):
    family = "Hermes-2"
    model_kwargs = {}
    header = ["name", "modelscope_name"]
    info = [
        # name
        ["NousResearch/Hermes-2-Pro-Llama-3-8B",    ""],
    ]


if __name__ == '__main__':
    import torch

    def transformers_test():
        from zerollama.microservices.inference.transformers_green.chat import run_test

        for model_name in ["NousResearch/Hermes-2-Pro-Llama-3-8B"]:
            run_test(model_name, stream=False)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name in ["NousResearch/Hermes-2-Pro-Llama-3-8B"]:
            run_test(model_name, stream=True)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

    transformers_test()