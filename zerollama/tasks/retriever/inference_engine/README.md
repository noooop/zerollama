# Retriever 推理引擎

### 帮助
```
$ python -m zerollama.tasks.retriever.cli --help
Usage: python -m zerollama.tasks.retriever.cli [OPTIONS] COMMAND [ARGS]...

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
$ python -m zerollama.tasks.retriever.cli list-families
Supported retriever families:
BGERetriever
```

### list-family 列出支持的模型家族成员
```
$ python -m zerollama.tasks.retriever.cli list-family BGERetriever
+--------------------------+-----------+-----------------+------------------------------------------------------------------------------------------------+--------+-----------+
| name                     | dimension | sequence_length | introduction                                                                                   | family | protocol  |
+--------------------------+-----------+-----------------+------------------------------------------------------------------------------------------------+--------+-----------+
| BAAI/bge-m3              | 1024      | 8192            | multilingual; unified fine-tuning (dense, sparse, and colbert) from bge-m3-unsupervised        | BGE    | retriever |
.....
+--------------------------+-----------+-----------------+------------------------------------------------------------------------------------------------+--------+-----------+
```

## 下载模型
```
$ python -m zerollama.tasks.retriever.cli pull BAAI/bge-m3
```


## demo 服务端启动
```
$ python -m zerollama.tasks.retriever.inference_engine.server
use inference backend:
zerollama.models.baai.backend.retriever:BGERetriever
ZeroNameServer: InMemoryNameServer running! port: 9527
ZeroRetrieverInferenceEngine:  BAAI/bge-m3 is running! port: 56812
```

## demo 客户端运行
```
$ python -m zerollama.tasks.retriever.inference_engine.client
================================================================================
Wait BAAI/bge-m3 available
[ServerInfo(name='BAAI/bge-m3', host='localhost', port=56812, protocol='retriever')]
================================================================================
ZeroRetrieverInferenceEngine support_methods
state='ok' msg={'support_methods': ['inference', 'info', 'support_methods']}
state='ok' msg={'name': 'BAAI/bge-m3', 'dimension': '1024', 'sequence_length': '8192', 'introduction': 'multilingual; unified fine-tuning (dense, sparse, and colbert) from bge-m3-unsupervised', 'family': 'BGERetriever', 'protocol': 'retriever'}
================================================================================
[[0.6265 0.3477]
 [0.35   0.678 ]]
```