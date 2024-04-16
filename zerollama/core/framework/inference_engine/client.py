
import zmq
import json
from zerollama.core.framework.nameserver.client import NameServerClient
from random import choice


class ChatClient(object):
    protocol = "chat"

    def __init__(self):
        context = zmq.Context()
        self.context = context
        self.nameserver_client = NameServerClient()

    def chat(self, name, messages, options=None):
        response = self.nameserver_client.get_services(self.protocol, name)

        if response["state"] == "error":
            return None

        services = response["msg"]["services"]
        if not services:
            return None

        server = choice(services)
        host = server["host"]
        port = server["port"]

        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{host}:{port}")

        data = json.dumps({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": messages,
            "options": options,
            "stream": False
        }).encode('utf8')

        socket.send(data)

        msg = socket.recv()
        msg = json.loads(msg)
        return msg

    def stream_chat(self, name, messages, options=None):
        response = self.nameserver_client.get_services(self.protocol, name)

        if response["state"] == "error":
            return None

        services = response["msg"]["services"]
        if not services:
            return None

        server = choice(services)
        host = server["host"]
        port = server["port"]

        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{host}:{port}")

        data = json.dumps({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": messages,
            "options": options,
            "stream": True
        }).encode('utf8')

        socket.send(data)

        part0 = socket.recv()
        yield json.loads(part0)
        while socket.getsockopt(zmq.RCVMORE):
            part = socket.recv()
            yield json.loads(part)


if __name__ == '__main__':
    from pprint import pprint

    client = ChatClient()
    print("=" * 80)
    print(client.nameserver_client.get_service_names(client.protocol))

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    name = "Qwen/Qwen1.5-0.5B-Chat"

    print("="*80)
    print("stream == False")
    msg = client.chat(name, messages)
    pprint(msg)

    print("="*80)
    print("stream == True")
    for msg in client.stream_chat(name, messages):
        pprint(msg)