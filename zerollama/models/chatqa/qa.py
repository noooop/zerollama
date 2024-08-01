from zerollama.tasks.qa.interface import QAModel


class ChatQA(QAModel):
    family = "ChatQA"
    model_kwargs = {}
    header = ["name", "modelscope_name"]
    info = [
        ["nvidia/Llama3-ChatQA-1.5-8B", "LLM-Research/Llama3-ChatQA-1.5-8B"],
        ["nvidia/Llama3-ChatQA-1.5-70B", "LLM-Research/Llama3-ChatQA-1.5-70B"],
    ]
    inference_backend = "zerollama.models.chatqa.backend.qa:ChatQA"


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor

    def transformers_test():
        import torch
        from zerollama.microservices.inference.transformers_green.chat import run_test

        for model_name in ["nvidia/Llama3-ChatQA-1.5-8B"]:
            print(model_name)
            run_test(model_name, stream=False)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name in ["nvidia/Llama3-ChatQA-1.5-8B"]:
            print(model_name)
            run_test(model_name, stream=True)
            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)


    with ThreadPoolExecutor(1) as executor:
        f = executor.submit(transformers_test)
        f.result()
