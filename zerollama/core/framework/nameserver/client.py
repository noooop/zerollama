
import time
from random import choice
from zerollama.core.framework.zero.client import Z_Client, Timeout
from zerollama.core.framework.nameserver.protocol import ServerInfo, GetServicesRequest, GetServiceNamesRequest

NameServerPort = 9527
CLIENT_VALIDATION = True


class NameServerClient(Z_Client):
    timeout = 100

    def __init__(self, port=None):
        if port is None:
            self.port = NameServerPort
        else:
            self.port = port
        Z_Client.__init__(self, f"tcp://localhost:{self.port}")

    def register(self, server_info):
        if CLIENT_VALIDATION:
            server_info = ServerInfo(**server_info).dict()

        data = {
            "method": "register",
            "data": server_info
        }
        return self.query(data)

    def deregister(self, server_info):
        if CLIENT_VALIDATION:
            server_info = ServerInfo(**server_info).dict()

        data = {
            "method": "deregister",
            "data": server_info
        }
        return self.query(data)

    def get_services(self, protocol, name):
        data = {"protocol": protocol, "name": name}

        if CLIENT_VALIDATION:
            data = GetServicesRequest(**data).dict()

        data = {
            "method": "get_services",
            "data": data
        }

        rep = self.query(data)
        if rep.state == "ok":
            rep.msg["services"] = [ServerInfo(**x) for x in rep.msg["services"]]
        return rep

    def get_service_names(self, protocol):
        data = {"protocol": protocol}

        if CLIENT_VALIDATION:
            data = GetServiceNamesRequest(**data).dict()

        data = {
            "method": "get_service_names",
            "data": data
        }
        return self.query(data)


class ZeroClient(object):
    def __init__(self, protocol, nameserver_port=None):
        self.protocol = protocol
        self.nameserver_client = NameServerClient(nameserver_port)

    def get_service_names(self):
        return self.nameserver_client.get_service_names(self.protocol)

    def get_services(self, name):
        rep = self.nameserver_client.get_services(self.protocol, name)

        if rep.state == "error":
            return None

        services = rep.msg["services"]
        if not services:
            return None

        return services

    def info(self, name):
        method = "info"
        client = self.get_client(name)
        if client is None:
            return None

        return self.query(name, method)

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
        client = Z_Client(f"tcp://{server.host}:{server.port}")
        return client

    def support_methods(self, name):
        method = "support_methods"
        client = self.get_client(name)
        if client is None:
            return None

        return self.query(name, method)

    def query(self, name, method, data=None, **kwargs):
        client = self.get_client(name)
        if client is None:
            return None
        if data is None:
            data = {}

        _data = {"method": method, "data": data}
        return client.query(_data, **kwargs)

    def stream_query(self, name, method, data=None, **kwargs):
        client = self.get_client(name)
        if client is None:
            return None
        if data is None:
            data = {}

        _data = {"method": method, "data": data}
        for part in client.stream_query(_data, **kwargs):
            yield part


if __name__ == '__main__':
    client = NameServerClient()
    CLIENT_VALIDATION = False
    name = "Qwen/Qwen1.5-0.5B-Chat"
    protocol = "chat"
    server_info = {"host": "localhost", "port": 9527, "name": name, "protocol": protocol}

    print("=" * 80)
    print('NameServer support_methods')
    print(client.support_methods())

    print("=" * 80)
    print('init')
    print(client.get_service_names(protocol))
    print(client.get_services(protocol, name))

    print("=" * 80)
    print("register")
    print(client.register(server_info))
    print(client.get_service_names(protocol))
    print(client.get_services(protocol, name))

    print("=" * 80)
    print("deregister")
    print(client.deregister(server_info))
    print(client.get_service_names(protocol))
    print(client.get_services(protocol, name))

