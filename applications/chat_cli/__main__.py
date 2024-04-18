

def pull(model_name):
    from zerollama.inference_backend.hf_transformers.download import download
    download(model_name)


def run(model_name):
    import time
    from zerollama.core.framework.zero.server import ZeroServerProcess

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer",
                                   server_kwargs={"port": "random"})
    nameserver.start()

    while True:
        time.sleep(0.1)
        if nameserver.share_port.value != -1:
            nameserver_port = nameserver.share_port.value
            break

    engine = ZeroServerProcess("zerollama.core.framework.inference_engine.server:ZeroInferenceEngine",
                               server_kwargs={
                                   "model_class": "zerollama.models.qwen.qwen1_5:Qwen1_5",
                                   "model_kwargs": {
                                     "model_name": model_name
                                   },
                                   "nameserver_port": nameserver_port
                               },
                               ignore_warnings=True)
    engine.start()

    from zerollama.core.framework.inference_engine.client import ChatClient
    chat_client = ChatClient(nameserver_port=nameserver_port)

    print("正在加载模型...")
    while True:
        time.sleep(0.1)
        response = chat_client.get_service_names()
        services = response["msg"]["service_names"]
        if model_name in services:
            break
    print("加载完成!")
    print("!quit 退出, !next 开启新一轮对话。玩的开心！")

    try:
        quit = False
        while not quit:
            print("=" * 80)
            messages = []
            i = 1
            while True:
                print(f"[对话第{i}轮]")
                prompt = input("(用户输入:)\n")

                if prompt == "!quit":
                    quit = True
                    break

                if prompt == "!next":
                    break

                messages.append({"role": "user", "content": prompt})

                print(f"({model_name}:)\n")
                content = ""
                for response in chat_client.stream_chat(model_name, messages):
                    print(response["content"], end="")
                    content += response["content"]
                print("\n")
                messages.append({"role": "assistant", "content": content})
                i += 1
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        engine.terminate()
        nameserver.terminate()
        print("quit gracefully")


def main(argv):
    method = argv[1]
    model_name = argv[2]

    if method == "pull":
        pull(model_name)

    if method == "run":
        run(model_name)


if __name__ == '__main__':
    import sys
    main(sys.argv)


