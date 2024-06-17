
from zerollama.tasks.chat.interface import ChatModel


class Index(ChatModel):
    family = "index"
    model_kwargs = {}
    header = ["name", "use_hf_only"]
    info = [
        # name
        ["IndexTeam/Index-1.9B-Chat",         True],
        ["IndexTeam/Index-1.9B-Character",    True],
    ]


if __name__ == '__main__':
    import torch
    from zerollama.microservices.inference.transformers_green.chat import run_test

    for model_name in ["IndexTeam/Index-1.9B-Chat"]:
        run_test(model_name, stream=False)

        print(torch.cuda.memory_allocated() / 1024 ** 2)

    for model_name in ["IndexTeam/Index-1.9B-Chat"]:
        run_test(model_name, stream=True)

        print(torch.cuda.memory_allocated() / 1024 ** 2)