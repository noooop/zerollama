
from multiprocess import Process


class HttpEntrypoint(Process):
    def __init__(self, name, engine_class, engine_kwargs=None, **kwargs):
        super().__init__()

        self.name = name
        self.engine_class = engine_class
        self.engine_kwargs = engine_kwargs or {}

    def init(self):
        pass

    def run(self):
        import uvicorn

        host = self.engine_kwargs.get("host", "0.0.0.0")
        port = self.engine_kwargs.get("port", 8000)

        self.engine_kwargs["host"] = host
        self.engine_kwargs["port"] = port

        print(f"HttpEntrypoints {self.name} running!", host, port)

        try:
            uvicorn.run(self.engine_class, **self.engine_kwargs)
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
    gateway = HttpEntrypoint(
        name="ollama_compatible",
        engine_class="zerollama.microservices.entrypoints.ollama_compatible.api:app",
        engine_kwargs={"port": 11434, "host": "127.0.0.1"})
    gateway.start()
    gateway.wait()

