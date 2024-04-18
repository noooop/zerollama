
import zmq
import json
from zerollama.core.framework.zero.client import Client
from zerollama.core.framework.nameserver.client import NameServerClient

from random import choice


class ChatClient(Client):
    protocol = "chat"

    def __init__(self, nameserver_port=None):
        context = zmq.Context()
        self.context = context
        self.nameserver_client = NameServerClient(nameserver_port)

    def get_service_names(self):
        return self.nameserver_client.get_service_names(self.protocol)

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

        data = json.dumps({
            "model": name,
            "messages": messages,
            "options": options,
            "stream": False
        }).encode('utf8')

        msg = self._query(addr=f"tcp://{host}:{port}",
                          data=data)
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

        data = json.dumps({
            "model": name,
            "messages": messages,
            "options": options,
            "stream": True
        }).encode('utf8')

        for part in self._stream_query(addr=f"tcp://{host}:{port}", data=data):
            yield json.loads(part)


if __name__ == '__main__':
    from pprint import pprint

    client = ChatClient()
    print("=" * 80)
    print(client.get_service_names())

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