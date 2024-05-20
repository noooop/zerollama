

def setup():
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.microservices.entrypoints.http_entrypoint import HttpEntrypoint

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    chat_manager = ZeroServerProcess("zerollama.core.framework.zero_manager.server:ZeroManager",
                                        server_kwargs={
                                            "name": "ZeroChatInferenceManager",
                                            "server_class": "zerollama.tasks.chat.engine.server:ZeroChatInferenceEngine"
                                        })
    retriever_manager = ZeroServerProcess("zerollama.core.framework.zero_manager.server:ZeroManager",
                                        server_kwargs={
                                            "name": "ZeroRetrieverInferenceManager",
                                            "server_class": "zerollama.tasks.retriever.engine.server:ZeroRetrieverInferenceEngine"
                                        })
    entrypoint1 = HttpEntrypoint(server_class="zerollama.microservices.entrypoints.ollama_compatible.api:app",
                                 server_kwargs={"port": 11434})

    entrypoint2 = HttpEntrypoint(server_class="zerollama.microservices.entrypoints.openai_compatible.api:app",
                                 server_kwargs={"port": 8080})

    handle = [nameserver, chat_manager, retriever_manager, entrypoint1, entrypoint2]
    return handle


def run(handle):
    for h in handle:
        h.start()

    for h in handle:
        h.wait()


if __name__ == '__main__':
    server = setup()
    run(server)

