from multiprocess import Process


class HttpGateway(Process):
    def run(self):
        import subprocess
        h = subprocess.Popen("python -m applications.chat_webui.http_gateway", shell=True)
        print("gateway running!")
        try:
            h.wait()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            print("HttpGateway clean_up!")

    def wait(self):
        pass


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess

    name = "ZeroInferenceManager"
    server_class = "zerollama.core.framework.inference_engine.server:ZeroInferenceEngine"

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    manager = ZeroServerProcess("zerollama.core.framework.zero_manager.server:ZeroManager",
                                server_kwargs={
                                    "name": name,
                                    "server_class": server_class
                                })
    gateway = HttpGateway()

    handle = [nameserver, manager, gateway]
    for h in handle:
        h.start()

    for h in handle:
        h.wait()

