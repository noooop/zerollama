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

        msg = socket.recv()
        msg = json.loads(msg)
        return msg


if __name__ == '__main__':
    from pprint import pprint

    client = Client()

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    result = client.query(port="9527",
                          messages=messages)

    pprint(result)