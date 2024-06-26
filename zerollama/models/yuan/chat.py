from zerollama.tasks.chat.interface import ChatModel, ChatGGUFModel


class Yuan(ChatModel):
    family = "Yuan"
    model_kwargs = {}
    header = ["name"]
    info = [
        # MoE
        ["IEITYuan/Yuan2-M32"],
        ["IEITYuan/Yuan2-M32-hf"]
    ]


class Yuan_GGUF(ChatGGUFModel):
    family = "Yuan_gguf"

    gguf = {
        "repo_id": [
            "IEITYuan/Yuan2-M32-gguf",
            "IEITYuan/Yuan2-M32-gguf-int4",
        ],
        "filename": [
            "*.gguf"
        ]
    }


if __name__ == '__main__':
    def llama_cpp_test():
        from zerollama.microservices.inference.llama_cpp_green.chat import run_test

        for model_name in ["IEITYuan/Yuan2-M32-gguf-int4+*.gguf"]:
            print(model_name)
            run_test(model_name, stream=False)

        for model_name in ["IEITYuan/Yuan2-M32-gguf-int4+*.gguf"]:
            print(model_name)
            run_test(model_name, stream=True)

    llama_cpp_test()