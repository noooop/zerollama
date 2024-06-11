
from zerollama.tasks.chat.cli import click, list_families, list_family, pull
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponseDone


@click.command()
@click.argument('model_name')
def run(model_name):
    from zerollama.core.framework.zero.server import ZeroServerProcess

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer",
                                   server_kwargs={"port": "random"})
    nameserver.start()
    nameserver_port = nameserver.wait_port_available()

    engine = ZeroServerProcess("zerollama.tasks.chat.engine.server:ZeroChatInferenceEngine",
                               server_kwargs={
                                   "name": model_name,
                                   "nameserver_port": nameserver_port
                               },
                               ignore_warnings=True)
    engine.start()

    from zerollama.tasks.chat.engine.client import ChatClient
    chat_client = ChatClient(nameserver_port=nameserver_port)

    print("正在加载模型...")
    chat_client.wait_service_available(model_name)
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

                print(f"({model_name}:)\n", flush=True)
                content = ""
                for rep in chat_client.stream_chat(model_name, messages):
                    if isinstance(rep, ChatCompletionStreamResponseDone):
                        print("\n", flush=True)
                        break
                    else:
                        print(rep.delta_content, end="", flush=True)
                        content += rep.delta_content

                messages.append({"role": "assistant", "content": content})
                i += 1
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        engine.terminate()
        nameserver.terminate()
        print("quit gracefully")


@click.group()
def chat():
    pass


chat.add_command(list_families)
chat.add_command(list_family)
chat.add_command(pull)
chat.add_command(run)


if __name__ == '__main__':
    chat()


