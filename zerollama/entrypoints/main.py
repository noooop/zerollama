
from multiprocess import Process


class HttpEntrypoint(Process):
    def __init__(self, server_class, server_kwargs=None):
        super().__init__()

        self.server_class = server_class
        self.server_kwargs = server_kwargs or {}

    def run(self):
        import uvicorn

        host = self.server_kwargs.get("host", "0.0.0.0")
        port = self.server_kwargs.get("port", 8000)

        self.server_kwargs["host"] = host
        self.server_kwargs["host"] = port

        print("HttpEntrypoints running!", host, port)

        try:
            uvicorn.run(self.server_class, **self.server_kwargs)
        except (KeyboardInterrupt, EOFError):
            pass
        print("HttpEntrypoints clean_up!")

    def wait(self):
        try:
            self.join()
        except (KeyboardInterrupt, EOFError):
            self.terminate()
        finally:
            self.terminate()


if __name__ == '__main__':
    gateway = HttpEntrypoint(server_class="zerollama.gateway.openai_compatible:app",
                             server_kwargs={"port": 11434, "host": "127.0.0.1"})
    gateway.start()
    gateway.wait()

