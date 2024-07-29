from zerollama.tasks.chat.interface import ChatModel


class Mistral(ChatModel):
    family = "Mistral"
    model_kwargs = {}
    header = ["name", "modelscope_name"]
    info = [
        # name
        ["mistralai/Mistral-7B-Instruct-v0.3",    "LLM-Research/Mistral-7B-Instruct-v0.3"],
        ["mistralai/Mistral-7B-Instruct-v0.2",    "LLM-Research/Mistral-7B-Instruct-v0.2"],
        ["mistralai/Mistral-7B-Instruct-v0.1",    "LLM-Research/Mistral-7B-Instruct-v0.1"],
        ["mistralai/Mixtral-8x7B-Instruct-v0.1",  "LLM-Research/Mixtral-8x7B-Instruct-v0.1"],
        ["mistralai/Mixtral-8x22B-Instruct-v0.1", "LLM-Research/Mixtral-8x22B-Instruct-v0.1"],
    ]


if __name__ == '__main__':
    import torch

    def transformers_test():
        from zerollama.microservices.inference.transformers_green.chat import run_test

        for model_name in ["mistralai/Mistral-7B-Instruct-v0.2"]:
            run_test(model_name, stream=False)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name in ["mistralai/Mistral-7B-Instruct-v0.2"]:
            run_test(model_name, stream=True)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

    transformers_test()