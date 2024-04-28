

def setup():
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.core.entrypoints.http_entrypoint import HttpEntrypoint

    name = "ZeroChatInferenceManager"
    server_class = "zerollama.tasks.chat.inference_engine.server:ZeroChatInferenceEngine"

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    manager = ZeroServerProcess("zerollama.core.framework.zero_manager.server:ZeroManager",
                                server_kwargs={
                                    "name": name,
                                    "server_class": server_class
                                })
    entrypoint1 = HttpEntrypoint(server_class="zerollama.tasks.chat.entrypoints.ollama_compatible.api:app",
                                 server_kwargs={"port": 11434})

    entrypoint2 = HttpEntrypoint(server_class="zerollama.tasks.chat.entrypoints.openai_compatible.api:app",
                                 server_kwargs={"port": 8080})

    handle = [nameserver, manager, entrypoint1, entrypoint2]
    return handle


def run(handle):
    for h in handle:
        h.start()

    for h in handle:
        h.wait()


if __name__ == '__main__':
    server = setup()
    run(server)

