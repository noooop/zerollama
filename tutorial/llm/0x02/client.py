import time
import zmq
import numpy as np
import json


class Client(object):
    def __init__(self):
        context = zmq.Context()
        self.context = context

    def query(self, port, messages):
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{port}")

        message = json.dumps({
            "model": "Qwen/Qwen1.5-0.5B",
            "messages": messages,
            "options": {
                "max_new_tokens": 128
            }
        }).encode('utf8')

        socket.send(message)

        msg = socket.recv_multipart()
        msg = [json.loads(m) for m in msg]
        if len(msg) == 1:
            msg = msg[0]
        return msg

    def stream_query(self, port, messages):
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{port}")

        message = json.dumps({
            "model": "Qwen/Qwen1.5-0.5B",
            "messages": messages,
            "options": {
                "max_new_tokens": 128
            }
        }).encode('utf8')

        socket.send(message)

        part1 = socket.recv()
        yield part1
        while socket.getsockopt(zmq.RCVMORE):
            part = socket.recv()
            yield part


if __name__ == '__main__':
    client = Client()

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    for msg in client.stream_query(port="9527", messages=messages):
        msg = json.loads(msg)
        print(msg)