from zerollama.core.framework.nameserver.server import InMemoryNameServer

nameserver = InMemoryNameServer()
name = "Qwen/Qwen1.5-0.5B-Chat"
protocol = "chat"
qwen_server = {"host": "localhost", "port": 9527, "name": name, "protocol": protocol}

print("="*80)
print('init')
print(nameserver.get_service_names(protocol))
print(nameserver.get_services(protocol, name))

print("="*80)
print("register")
nameserver.register(qwen_server)
print(nameserver.get_service_names(protocol))
print(nameserver.get_services(protocol, name))

print("="*80)
print("deregister")
nameserver.deregister(qwen_server)
print(nameserver.get_service_names(protocol))
print(nameserver.get_services(protocol, name))



