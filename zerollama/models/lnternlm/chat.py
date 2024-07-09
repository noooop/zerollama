from zerollama.tasks.chat.interface import ChatModel, ChatGGUFModel


class InternLM(ChatModel):
    family = "InternLM"
    model_kwargs = {"trust_remote_code": True}
    header = ["name", "modelscope_name", "size", "quantization", "bits"]
    info = [
        #name                                     modelscope_name                                    size      quantization(_, GPTQ, AWQ)     bits
        ["internlm/internlm-chat-7b",             "Shanghai_AI_Laboratory/internlm-chat-7b",         "7b",     "",                            ""],
        ["internlm/internlm-chat-20b",            "Shanghai_AI_Laboratory/internlm-chat-20b",        "20b",    "",                            ""],
        ["internlm/internlm-chat-20b-4bit",       "",                                                "20b",    "AWQ",                         "4bit"],

        ["internlm/internlm2-chat-1_8b",          "Shanghai_AI_Laboratory/internlm2-chat-1_8b",      "1.8b",   "",                            ""],
        ["internlm/internlm2-chat-7b",            "Shanghai_AI_Laboratory/internlm2-chat-7b",        "7b",     "",                            ""],
        ["internlm/internlm2-chat-20b",           "Shanghai_AI_Laboratory/internlm2-chat-20b",       "20b",    "",                            ""],
        ["internlm/internlm2-chat-7b-4bit",       "",                                                "7b",     "AWQ",                         "4bit"],
        ["internlm/internlm2-chat-20b-4bit",      "",                                                "20b",    "AWQ",                         "4bit"],

        ["internlm/internlm2_5-7b-chat",          "Shanghai_AI_Laboratory/internlm2_5-7b-chat",      "7b",     "",                            ""],
        ["internlm/internlm2_5-7b-chat-1m",       "Shanghai_AI_Laboratory/internlm2_5-7b-chat-1m",   "7b",     "",                            ""],
        ["internlm/internlm2_5-7b-chat-4bit",     "",                                                "7b",     "AWQ",                         "4bit"],
    ]


class InternLM_GGUF(ChatGGUFModel):
    family = "Qwen1.5_gguf"

    gguf = {
        "repo_id": [
            "internlm/internlm2_5-7b-chat-gguf"
        ],
        "filename": [
            "*fp16.gguf",
            "*q8_0.gguf",
            "*q6_k.gguf",
            "*q5_k_m.gguf",
            "*q5_0.gguf",
            "*q4_k_m.gguf",
            "*q4_0.gguf",
            "*q3_k_m.gguf",
            "*q2_k.gguf"
        ]
    }


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor

    def transformers_test():
        import torch
        from zerollama.microservices.inference.transformers_green.chat import run_test

        for model_name in ["internlm/internlm2_5-7b-chat"]:
            print(model_name)
            run_test(model_name, stream=False)

            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)

        for model_name in ["internlm/internlm2_5-7b-chat"]:
            print(model_name)
            run_test(model_name, stream=True)
            print("memory_allocated:", torch.cuda.memory_allocated() / 1024 ** 2)


    def llama_cpp_test():
        from zerollama.microservices.inference.llama_cpp_green.chat import run_test

        for model_name in ["internlm/internlm2_5-7b-chat-gguf+*q4_k_m.gguf"]:
            run_test(model_name, stream=False)

        for model_name in ["internlm/internlm2_5-7b-chat-gguf+*q4_k_m.gguf"]:
            run_test(model_name, stream=True)


    with ThreadPoolExecutor(1) as executor:
        f = executor.submit(transformers_test)
        f.result()

    with ThreadPoolExecutor(1) as executor:
        f = executor.submit(llama_cpp_test)
        f.result()