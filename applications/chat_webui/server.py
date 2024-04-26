

def setup():
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.entrypoints.main import HttpEntrypoint

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    manager = ZeroServerProcess("zerollama.core.framework.zero_manager.server:ZeroManager",
                                server_kwargs={
                                    "name": "ZeroInferenceManager",
                                    "server_class": "zerollama.core.framework.inference_engine.server:ZeroInferenceEngine"
                                })
    entrypoint1 = HttpEntrypoint(server_class="zerollama.entrypoints.ollama_compatible.api:app",
                                 server_kwargs={"port": 11434})

    entrypoint2 = HttpEntrypoint(server_class="zerollama.entrypoints.openai_compatible.api:app",
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

