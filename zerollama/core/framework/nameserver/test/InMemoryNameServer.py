from zerollama.core.framework.nameserver.server import InMemoryNameServer
from zerollama.core.framework.nameserver.protocol import ServerInfo

nameserver = InMemoryNameServer()
name = "Qwen/Qwen1.5-0.5B-Chat"
protocol = "chat"
server_info = ServerInfo(**{"host": "localhost", "port": 9527, "name": name, "protocol": protocol})

print("="*80)
print('init')
print(nameserver.get_service_names(protocol))
print(nameserver.get_services(protocol, name))

print("="*80)
print("register")
nameserver.register(server_info)
print(nameserver.get_service_names(protocol))
print(nameserver.get_services(protocol, name))

print("="*80)
print("deregister")
nameserver.deregister(server_info)
print(nameserver.get_service_names(protocol))
print(nameserver.get_services(protocol, name))



