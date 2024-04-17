
import zmq
import json
from zerollama.core.framework.zeroserver.server import ZeroServer

NameServerPort = 9527


class NameServerInterfaces(object):
    def register(self, server):
        raise NotImplementedError

    def deregister(self, server):
        # https://english.stackexchange.com/questions/25931/unregister-vs-deregister
        raise NotImplementedError

    def get_services(self, protocol, name):
        raise NotImplementedError

    def get_service_names(self, protocol):
        raise NotImplementedError


class InMemoryNameServer(NameServerInterfaces):
    def __init__(self):
        self.domain = dict()

    def register(self, server):
        name = server["name"]
        protocol = server["protocol"]

        if protocol not in self.domain:
            self.domain[protocol] = {}

        if name not in self.domain[protocol]:
            self.domain[protocol][name] = []

        self.domain[protocol][name].append(server)

    def deregister(self, server):
        name = server["name"]
        protocol = server["protocol"]

        if protocol not in self.domain:
            return False

        if name not in self.domain[protocol]:
            return False

        services = self.domain[protocol][name]

        if server not in services:
            return False

        services = [s for s in services if s != server]
        if len(services) > 0:
            self.domain[protocol][name] = services
        else:
            del self.domain[protocol][name]

        return True

    def get_services(self, protocol, name):
        services = self.domain.get(protocol, dict()).get(name, None)
        return services

    def get_service_names(self, protocol):
        return list(self.domain.get(protocol, dict()).keys())


class ZeroNameServer(ZeroServer):
    def __init__(self, nameserver_class=None, event=None):
        ZeroServer.__init__(self, port=NameServerPort, event=event, do_register=False)

        if nameserver_class is None:
            self.nameserver_class = InMemoryNameServer
        else:
            self.nameserver_class = nameserver_class

        self._nameserver = None

    def init(self):
        self._nameserver = self.nameserver_class()
        print(f"ZeroNameServer: {self.nameserver_class.__name__} running!")

    def process(self):
        msg = self.socket.recv()
        try:
            msg = json.loads(msg)

            if "method" not in msg:
                self.handle_error(err_msg="'method' not in msg")
                return

            method = msg["method"]

            if method == "register":
                self.register(msg)
                return

            if method == "deregister" or method == "unregister":
                self.deregister(msg)
                return

            if method == "get_services":
                self.get_services(msg)
                return

            if method == "get_service_names":
                self.get_service_names(msg)
                return

            self.handle_error(err_msg=f"method [{method}] not supported.")
        except Exception:
            self.handle_error(err_msg="NameServer error")

    def clean(self, msg):
        ok = False

        if "server" not in msg:
            err_msg = "'server' not in msg"
            return ok, err_msg

        server = msg["server"]

        if "name" not in server:
            err_msg = "'name' not in msg.server"
            return ok, err_msg

        name = server["name"]

        if "host" not in server:
            err_msg = "'host' not in msg.server"
            return ok, err_msg

        host = server["host"]

        if "port" not in server:
            err_msg = "'port' not in msg.server"
            return ok, err_msg

        port = server["port"]

        try:
            port = int(port)
        except ValueError:
            err_msg = f"'port' must be int, [{port}] founded."
            return ok, err_msg

        if 'protocol' not in server:
            err_msg = "'protocol' not in msg.server"
            return ok, err_msg

        protocol = server["protocol"]

        ok = True
        server = {
            "name": name,
            "host": host,
            "port": port,
            "protocol": protocol
        }

        return ok, server

    def register(self, msg):
        ok, out = self.clean(msg)
        if not ok:
            self.handle_error(err_msg=out)
            return

        server = out
        self._nameserver.register(server)

        response = json.dumps({
            "state": "ok",
            "msg": "register success"
        }).encode('utf8')

        self.socket.send(response)

    def deregister(self, msg):
        ok, out = self.clean(msg)
        if not ok:
            self.handle_error(err_msg=out)
            return

        server = out
        founded = self._nameserver.deregister(server)

        response = json.dumps({
            "state": "ok",
            "msg": {"founded": founded}
        }).encode('utf8')

        self.socket.send(response)

    def get_services(self, msg):
        if "protocol" not in msg:
            err_msg = "'protocol' not in msg"
            self.handle_error(err_msg)
            return
        protocol = msg['protocol']

        if "name" not in msg:
            err_msg = "'name' not in msg"
            self.handle_error(err_msg)
            return

        name = msg['name']

        services = self._nameserver.get_services(protocol, name)
        response = json.dumps({
            "state": "ok",
            "msg": {"services": services}
        }).encode('utf8')
        self.socket.send(response)

    def get_service_names(self, msg):
        if "protocol" not in msg:
            err_msg = "'protocol' not in msg"
            self.handle_error(err_msg)
            return

        protocol = msg['protocol']

        service_names = self._nameserver.get_service_names(protocol)
        response = json.dumps({
            "state": "ok",
            "msg": {"service_names": service_names}
        }).encode('utf8')
        self.socket.send(response)


def nameserver(nameserver_class=None):
    nameserver_class = nameserver_class or InMemoryNameServer
    server = ZeroNameServer(nameserver_class=nameserver_class)
    server.run()


if __name__ == '__main__':
    nameserver()
