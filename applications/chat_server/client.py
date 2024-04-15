import time
import zmq
import numpy as np
import json


class Client(object):
    def __init__(self):
        context = zmq.Context()
        self.context = context

    def query(self, port, data):
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{port}")

        socket.send(data)

        msg = socket.recv()
        msg = json.loads(msg)
        return msg

    def stream_query(self, port, data):
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{port}")

        socket.send(data)

        part0 = socket.recv()
        yield json.loads(part0)
        while socket.getsockopt(zmq.RCVMORE):
            part = socket.recv()
            yield json.loads(part)


if __name__ == '__main__':
    from pprint import pprint

    PORT = 9527

    client = Client()

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    print("="*80)
    print("stream == False")
    data = json.dumps({
        "model": "Qwen/Qwen1.5-0.5B-Chat",
        "messages": messages,
        "options": {"max_new_tokens": 512},
        "stream": False
    }).encode('utf8')

    msg = client.query(port=PORT, data=data)
    pprint(msg)

    print("="*80)
    print("stream == True")
    data = json.dumps({
        "model": "Qwen/Qwen1.5-0.5B-Chat",
        "messages": messages,
        "options": {"max_new_tokens": 512},
        "stream": True
    }).encode('utf8')

    for msg in client.stream_query(port=PORT, data=data):
        pprint(msg)