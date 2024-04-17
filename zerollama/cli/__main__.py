

def pull(model_name):
    from zerollama.inference_backend.hf_transformers.download import download
    download(model_name)


def run_nameserver():
    import warnings
    warnings.filterwarnings("ignore")

    from zerollama.core.framework.nameserver.server import nameserver
    nameserver()


def run_model_server(model_name):
    import warnings
    warnings.filterwarnings("ignore")

    from zerollama.core.framework.inference_engine.server import ZeroInferenceEngine
    from zerollama.models.qwen.qwen1_5 import Qwen1_5

    engine = ZeroInferenceEngine(model_class=Qwen1_5,
                                 model_kwargs={
                                     "model_name": model_name
                                 })
    print("ZeroInferenceEngine: ", engine.name, "is running!")
    engine.run()


def run(model_name):
    import time

    import warnings
    warnings.filterwarnings("ignore")

    from multiprocess import Process
    h1 = Process(target=run_model_server, args=(model_name,))
    h2 = Process(target=run_nameserver)

    handle = [h1, h2]
    for h in handle:
        h.start()

    from zerollama.core.framework.inference_engine.client import ChatClient
    chat_client = ChatClient()

    print("正在加载模型...")
    while True:
        time.sleep(0.1)
        response = chat_client.get_service_names()
        services = response["msg"]["service_names"]
        if model_name in services:
            break
    print("加载完成!")

    messages = []
    i = 1
    while True:
        print(f"[对话第{i}轮]")
        prompt = input("(用户输入:)\n")
        messages.append({"role": "user", "content": prompt})

        print(f"({model_name}:)\n")
        content = ""
        for response in chat_client.stream_chat(model_name, messages):
            print(response["content"], end="")
            content += response["content"]
        print("\n")
        messages.append({"role": "assistant", "content": content})
        i += 1


def main(argv):
    method = argv[1]
    model_name = argv[2]

    if method == "pull":
        pull(model_name)

    if method == "run":
        run(model_name)


if __name__ == '__main__':
    import sys
    argv = sys.argv
    main(sys.argv)


