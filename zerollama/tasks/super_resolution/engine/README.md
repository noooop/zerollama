# super resolution 

### 帮助
```
$ python -m zerollama.tasks.super_resolution.cli
Usage: python -m zerollama.tasks.super_resolution.cli [OPTIONS] COMMAND
                                                      [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  list-families
  list-family
  pull
  start
  terminate
```

### list-families 列出支持的模型家族
```
$ python -m zerollama.tasks.super_resolution.cli list-families
Supported retriever families:
APISR
```

### list-family 列出支持的模型家族成员
```
$ python -m zerollama.tasks.super_resolution.cli list-family APISR
+--------+-------+---------------------------------+--------+------------------+
| name   | scale | weight_path                     | family | protocol         |
+--------+-------+---------------------------------+--------+------------------+
| 4xGRL  | 4     | 4x_APISR_GRL_GAN_generator.pth  | APISR  | super_resolution |
| 4xDAT  | 4     | 4x_APISR_DAT_GAN_generator.pth  | APISR  | super_resolution |
| 4xRRDB | 4     | 4x_APISR_RRDB_GAN_generator.pth | APISR  | super_resolution |
| 2xRRDB | 2     | 2x_APISR_RRDB_GAN_generator.pth | APISR  | super_resolution |
+--------+-------+---------------------------------+--------+------------------+
```

## demo 服务端启动
```
$ python -m zerollama.tasks.super_resolution.engine.server
use inference backend:
zerollama.models.apisr.backend.sr:APISR
ZeroNameServer: InMemoryNameServer running! port: 9527
Number of parameters  1.03
ZeroSRInferenceEngine:  4xGRL is running! port: 54965
```

## demo 客户端运行
```
$ python -m zerollama.tasks.super_resolution.engine.client
================================================================================
Wait 4xGRL available
[ServerInfo(name='4xGRL', host='localhost', port=54965, protocol='super_resolution')]
================================================================================
ZeroSRInferenceEngine support_methods
state='ok' msg={'name': 'ZeroSRInferenceEngine', 'support_methods': ['inference', 'info', 'support_methods']}
state='ok' msg={}
input shape (620, 878, 3), output shape (2480, 3504, 3)
```