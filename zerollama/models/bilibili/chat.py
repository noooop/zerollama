
from zerollama.tasks.chat.interface import ChatModel, ChatGGUFModel


class Index(ChatModel):
    family = "index"
    model_kwargs = {"trust_remote_code": True}
    header = ["name"]
    info = [
        # name
        ["IndexTeam/Index-1.9B-Chat"],
        ["IndexTeam/Index-1.9B-Character"],
    ]


class IndexGGUF(ChatGGUFModel):
    family = "index_gguf"

    gguf = {
        "repo_id": [
            "IndexTeam/Index-1.9B-Chat-GGUF",
        ],
        "filename": [
            "*bf16.gguf",
            "*Q8_0.gguf",
            "*Q6_K.gguf",
            "*Q4_K_M.gguf",
            "*Q4_0.gguf"
        ]
    }


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor

    def transformers_test():
        import torch
        from zerollama.microservices.inference.transformers_green.chat import run_test

        for model_name in ["IndexTeam/Index-1.9B-Chat"]:
            print(model_name)
            run_test(model_name, stream=False)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name in ["IndexTeam/Index-1.9B-Chat"]:
            print(model_name)
            run_test(model_name, stream=True)
            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

    def llama_cpp_test():
        from zerollama.microservices.inference.llama_cpp_green.chat import run_test

        for model_name in ["IndexTeam/Index-1.9B-Chat-GGUF+*Q4_0.gguf"]:
            run_test(model_name, stream=False)

        for model_name in ["IndexTeam/Index-1.9B-Chat-GGUF+*Q4_0.gguf"]:
            run_test(model_name, stream=True)

    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(transformers_test)
        f.result()

    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(llama_cpp_test)
        f.result()

