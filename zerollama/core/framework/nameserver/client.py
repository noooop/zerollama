
import time
from random import choice
from zerollama.core.framework.zero.client import Z_Client, Timeout

NameServerPort = 9527


class NameServerClient(Z_Client):
    timeout = 100

    def __init__(self, port=None):
        if port is None:
            self.port = NameServerPort
        else:
            self.port = port
        Z_Client.__init__(self, f"tcp://localhost:{self.port}")

    def register(self, server):
        data = {
            "method": "register",
            "server": server
        }
        return self.json_query(data)

    def deregister(self, server):
        data = {
            "method": "deregister",
            "server": server
        }
        return self.json_query(data)

    def get_services(self, protocol, name):
        data = {
            "method": "get_services",
            "protocol": protocol,
            "name": name,
        }
        return self.json_query(data)

    def get_service_names(self, protocol):
        data = {
            "method": "get_service_names",
            "protocol": protocol,
        }
        return self.json_query(data)


class ZeroClient(object):
    def __init__(self, protocol, nameserver_port=None):
        self.protocol = protocol
        self.nameserver_client = NameServerClient(nameserver_port)

    def get_service_names(self):
        return self.nameserver_client.get_service_names(self.protocol)

    def get_services(self, name):
        response = self.nameserver_client.get_services(self.protocol, name)

        if response["state"] == "error":
            return None

        services = response["msg"]["services"]
        if not services:
            return None

        return services

    def wait_service_available(self, name, timeout=10000):
        t = timeout + time.time()

        while time.time() < t:
            time.sleep(0.1)
            try:
                services = self.get_services(name)
                if services:
                    return services
            except Timeout:
                pass

        raise Timeout

    def get_client(self, name):
        services = self.get_services(name)
        if not services:
            return None

        server = choice(services)
        host = server["host"]
        port = server["port"]

        client = Z_Client(f"tcp://{host}:{port}")
        return client

    def support_methods(self, name):
        client = self.get_client(name)
        if client is None:
            return None

        data = {"method": "support_methods"}
        return self.json_query(name, data)

    def json_query(self, name, data, **kwargs):
        client = self.get_client(name)
        if client is None:
            return None

        return client.json_query(data, **kwargs)

    def json_stream_query(self, name, data, **kwargs):
        client = self.get_client(name)
        if client is None:
            return None

        for part in client.json_stream_query(data, **kwargs):
            yield part


if __name__ == '__main__':
    client = NameServerClient()
    name = "Qwen/Qwen1.5-0.5B-Chat"
    protocol = "chat"
    qwen_server = {"host": "localhost", "port": 9527, "name": name, "protocol": protocol}

    print("=" * 80)
    print('NameServer support_methods')
    print(client.support_methods())

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

