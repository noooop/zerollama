

if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.entrypoints.main import HttpEntrypoint

    name = "ZeroInferenceManager"
    server_class = "zerollama.core.framework.inference_engine.server:ZeroInferenceEngine"

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    manager = ZeroServerProcess("zerollama.core.framework.zero_manager.server:ZeroManager",
                                server_kwargs={
                                    "name": name,
                                    "server_class": server_class
                                })
    entrypoint1 = HttpEntrypoint(server_class="zerollama.entrypoints.ollama_compatible.api:app",
                                 server_kwargs={"port": 11434})

    entrypoint2 = HttpEntrypoint(server_class="zerollama.entrypoints.openai_compatible.api:app",
                                 server_kwargs={"port": 8080})

    handle = [nameserver, manager, entrypoint1, entrypoint2]
    for h in handle:
        h.start()

    for h in handle:
        h.wait()

