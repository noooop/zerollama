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

            dtime = datetime.datetime.now()

            created_at = dtime.strftime("%Y-%m-%dT%H:%M:%S")
            msg = json.loads(msg)

            model = msg["model"]
            messages = msg["messages"]
            options = msg.get("options", dict())

            result = self.model.chat(messages, options)

            response = json.dumps({
                "model": model,
                "created_at": created_at,
                "message": {
                    "role": "assistant",
                    "content": result["response_text"],
                    "images": None
                },
                "done": True
            }).encode('utf8')

            self.socket.send(response)


if __name__ == '__main__':
    from model import Qwen

    server = Server(port=9527,
                    model_class=Qwen,
                    model_kwargs={
                        "model_name": "Qwen/Qwen1.5-0.5B"
                    })

    server.run()