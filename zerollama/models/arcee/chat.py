from zerollama.tasks.chat.interface import ChatModel, ChatGGUFModel


class ArceeSpark(ChatModel):
    family = "Arcee Spark"
    header = ["name", "use_hf_only"]
    info = [
        ["arcee-ai/Arcee-Spark", True]
    ]


if __name__ == '__main__':
    def transformers_test():
        import torch
        from zerollama.microservices.inference.transformers_green.chat import run_test

        for model_name in ["arcee-ai/Arcee-Spark"]:
            print(model_name)
            run_test(model_name, stream=False)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name in ["arcee-ai/Arcee-Spark"]:
            print(model_name)
            run_test(model_name, stream=True)
            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)


    transformers_test()