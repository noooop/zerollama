import zmq
import json
import datetime


class Server(object):
    def __init__(self, port, model_class, model_kwargs):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{port}")
        
        self.context = context
        self.socket = socket
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.model = self.model_class(**self.model_kwargs)
    
    def run(self):
        self.model.load()
        
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
    from zerollama.models.qwen.qwen1_5 import Qwen1_5

    server = Server(port=9527,
                    model_class=Qwen1_5,
                    model_kwargs={
                        "model_name": "Qwen/Qwen1.5-0.5B-Chat"
                    })

    server.run()