import zmq
import json
import datetime


class ZeroInferenceEngine(object):
    def __init__(self, model_class, model_kwargs):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        port = socket.bind_to_random_port("tcp://*", min_port=50000, max_port=60000)

        self.port = port
        self.name = model_kwargs["model_name"]
        self.protocol = model_class.protocol

        self.context = context
        self.socket = socket
        self.model_class = model_class
        self.model_kwargs = model_kwargs

        self.model = self.model_class(**self.model_kwargs)

    def register(self):
        from zerollama.core.framework.nameserver.client import NameServerClient
        client = NameServerClient()

        server = {"host": "localhost", "port": self.port, "name": self.name, "protocol": self.protocol}
        client.register(server)

    def run(self):
        self.model.load()
        self.register()

        while True:
            msg = self.socket.recv()
            msg = json.loads(msg)

            if "model" not in msg:
                response = json.dumps({"error": "model is required"}).encode('utf8')
                self.socket.send(response)
                continue

            model = msg["model"]

            if model != self.model.model_name:
                response = json.dumps({"error": f"model '{model}' not supported!"}).encode('utf8')
                self.socket.send(response)
                continue

            messages = msg.get("messages", list())
            options = msg.get("options", dict())
            stream = msg.get("stream", True)

            if stream:
                try:
                    for content in self.model.stream_chat(messages, options):
                        response = json.dumps({
                            "model": model,
                            "content": content,
                            "done": False,
                        }).encode('utf8')
                        self.socket.send(response, zmq.SNDMORE)
                finally:
                    response = json.dumps({
                        "model": model,
                        "content": "",
                        "done": True,
                    }).encode('utf8')
                    self.socket.send(response)
            else:
                content = self.model.chat(messages, options)
                response = json.dumps({
                    "model": model,
                    "content": content,
                }).encode('utf8')

                self.socket.send(response)


if __name__ == '__main__':
    from multiprocess import Process

    def run_nameserver():
        from zerollama.core.framework.nameserver.server import nameserver
        nameserver()

    h = Process(target=run_nameserver)
    h.start()

    from zerollama.models.qwen.qwen1_5 import Qwen1_5

    engine = ZeroInferenceEngine(model_class=Qwen1_5,
                                 model_kwargs={
                                     "model_name": "Qwen/Qwen1.5-0.5B-Chat"
                                 })
    print("ZeroInferenceEngine: ", engine.name, "is running!")
    engine.run()
    h.join()