from zerollama.tasks.chat.interface import ChatModel


class GLM4(ChatModel):
    family = "GLM-4"
    model_kwargs = {"trust_remote_code": True}
    header = ["name", "modelscope_name", "Seq Length"]
    info = [
        # name                                   modelscope_name              Seq Length
        ["THUDM/glm-4-9b-chat",                  "ZhipuAI/glm-4-9b-chat",     "128K"],
        ["THUDM/glm-4-9b-chat-1m",               "ZhipuAI/glm-4-9b-chat-1m",  "1M"],
    ]


if __name__ == '__main__':
    import torch

    def transformers_test():
        from zerollama.microservices.inference.transformers_green.chat import run_test
        from transformers import BitsAndBytesConfig

        for model_name, kwargs in [("THUDM/glm-4-9b-chat", {})]:
            print(model_name)
            run_test(model_name, stream=False, **kwargs)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name, kwargs in [("THUDM/glm-4-9b-chat", {})]:
            print(model_name)
            run_test(model_name, stream=True, **kwargs)
            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

    transformers_test()