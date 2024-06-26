

def setup():
    from zerollama.core.framework.zero.server import ZeroServerProcess
    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    manager = ZeroServerProcess("zerollama.core.framework.zero_manager.server:ZeroManager",
                                server_kwargs={
                                    "name": "RootZeroManager",
                                    "server_class": "zerollama.core.framework.zero_manager.server:ZeroManager"
                                })
    return nameserver, manager


def run(handle, waiting=True):
    nameserver, manager = handle
    nameserver.start()
    nameserver.wait_port_available()

    if nameserver.status != "running":
        return

    manager.start()

    if waiting:
        for h in handle:
            h.wait()


def start():
    server = setup()
    run(server)


if __name__ == '__main__':
    start()

