# 服务发现
如果不想在代码里面写死每个模型，或者叫微服务的服务器地址端口(host:port), 以及能提供的服务，就需要一个服务注册中心。

## NameServerInterfaces 接口定义
1. register 注册服务
2. deregister 取消注册
3. get_services 通过 protocol 和 name 查询服务
4. get_service_names 通过 protocol 查询服务名

## InMemoryNameServer
```
$ python -m zerollama.core.framework.nameserver.test.InMemoryNameServer
================================================================================
init
[]
None
================================================================================
register
['Qwen/Qwen1.5-0.5B-Chat']
[{'host': 'localhost', 'port': 9527, 'name': 'Qwen/Qwen1.5-0.5B-Chat', 'protocol': 'chat'}]
================================================================================
deregister
[]
None
```

## NameServer客户端
```
第一个窗口
$ python -m zerollama.core.framework.nameserver.server
ZeroNameServer: InMemoryNameServer running! port: 9527

第二个窗口
$ python -m zerollama.core.framework.nameserver.client
================================================================================
NameServer support_methods
{'state': 'ok', 'support_methods': ['deregister', 'get_service_names', 'get_services', 'register', 'support_methods']}
================================================================================
init
{'state': 'ok', 'msg': {'service_names': []}}
{'state': 'ok', 'msg': {'services': None}}
================================================================================
register
{'state': 'ok', 'msg': {'service_names': ['Qwen/Qwen1.5-0.5B-Chat']}}
{'state': 'ok', 'msg': {'services': [{'name': 'Qwen/Qwen1.5-0.5B-Chat', 'host': 'localhost', 'port': 9527, 'protocol': 'chat'}]}}
================================================================================
deregister
{'state': 'ok', 'msg': {'service_names': []}}
{'state': 'ok', 'msg': {'services': None}}

# 第一个窗口 Control-C 关闭 
ZeroNameServer clean_up!
```