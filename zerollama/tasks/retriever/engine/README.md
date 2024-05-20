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
bge-retrieval
bce-embedding
m3e
```

### list-family 列出支持的模型家族成员
```
$ python -m python -m zerollama.tasks.retriever.cli list-family m3e
+-------------------+------+-----------+----+----+-----+-----+-----+-------------+---------------+--------+-----------+
| name              | size | dimension | zh | en | s2s | s2p | s2c | open source | compatibility | family | protocol  |
+-------------------+------+-----------+----+----+-----+-----+-----+-------------+---------------+--------+-----------+
| moka-ai/m3e-small | 24M  | 512       | 是 | 否 | 是  | 否  | 否  | 是          | 优            | m3e    | retriever |
| moka-ai/m3e-base  | 110M | 768       | 是 | 是 | 是  | 是  | 否  | 是          | 优            | m3e    | retriever |
| moka-ai/m3e-large | 340M | 768       | 是 | 否 | 是  | 是  | 否  | 是          | 优            | m3e    | retriever |
+-------------------+------+-----------+----+----+-----+-----+-----+-------------+---------------+--------+-----------+
```

## 下载模型
```
$ python -m zerollama.tasks.retriever.cli pull BAAI/bge-m3
```


## demo 服务端启动
```
$ python -m zerollama.tasks.retriever.engine.server
use inference backend:
zerollama.models.baai.backend.retriever:BGERetriever
ZeroNameServer: InMemoryNameServer running! port: 9527
ZeroRetrieverInferenceEngine:  BAAI/bge-m3 is running! port: 56539
```

## demo 客户端运行
```
$ python -m zerollama.tasks.retriever.engine.client
================================================================================
Wait BAAI/bge-m3 available
[ServerInfo(name='BAAI/bge-m3', host='localhost', port=56539, protocol='retriever')]
================================================================================
ZeroRetrieverInferenceEngine support_methods
state='ok' msg={'name': 'ZeroRetrieverInferenceEngine', 'support_methods': ['inference', 'info', 'support_methods']}
state='ok' msg={'name': 'BAAI/bge-m3', 'dimension': '1024', 'sequence_length': '8192', 'introduction': 'multilingual; unified fine-tuning (dense, sparse, and colbert) from bge-m3-unsupervised', 'family': 'bge-retrieval', 'protocol': 'retriever'}
================================================================================
[[0.6265 0.3477]
 [0.35   0.678 ]]
```