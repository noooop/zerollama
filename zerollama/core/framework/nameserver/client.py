
import zmq
import json


NameServerPort = 9527


class NameServerClient(object):
    def __init__(self):
        context = zmq.Context()
        self.context = context

    def query(self, data):
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{NameServerPort}")

        data = json.dumps(data).encode('utf8')
        socket.send(data)

        msg = socket.recv()
        msg = json.loads(msg)
        return msg

    def register(self, server):
        data = {
            "method": "register",
            "server": server
        }
        return self.query(data)

    def deregister(self, server):
        data = {
            "method": "deregister",
            "server": server
        }
        return self.query(data)

    def get_services(self, protocol, name):
        data = {
            "method": "get_services",
            "protocol": protocol,
            "name": name,
        }
        return self.query(data)

    def get_service_names(self, protocol):
        data = {
            "method": "get_service_names",
            "protocol": protocol,
        }
        return self.query(data)


if __name__ == '__main__':
    client = NameServerClient()
    name = "Qwen/Qwen1.5-0.5B-Chat"
    protocol = "chat"
    qwen_server = {"host": "localhost", "port": 9527, "name": name, "protocol": protocol}

    print("=" * 80)
    print('init')
    print(client.get_service_names(protocol))
    print(client.get_services(protocol, name))

    print("=" * 80)
    print("register")
    client.register(qwen_server)
    print(client.get_service_names(protocol))
    print(client.get_services(protocol, name))

    print("=" * 80)
    print("deregister")
    client.deregister(qwen_server)
    print(client.get_service_names(protocol))
    print(client.get_services(protocol, name))

