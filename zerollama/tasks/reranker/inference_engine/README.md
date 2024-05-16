# Reranker 推理引擎

### 帮助
```
$ python -m zerollama.tasks.reranker.cli --help
Usage: python -m zerollama.tasks.reranker.cli [OPTIONS] COMMAND [ARGS]...

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
$ python -m zerollama.tasks.reranker.cli list-families
Supported retriever families:
BGEReranker
```

### list-family 列出支持的模型家族成员
```
$ python -m zerollama.tasks.reranker.cli list-family BGEReranker
+----------------------------------------+-----------+-----------+---------------------+---------------------+------------------------------------------------------------------------+-------------+----------+
| name                                   | type      | layerwise | Base model          | Language            | feature                                                                | family      | protocol |
+----------------------------------------+-----------+-----------+---------------------+---------------------+------------------------------------------------------------------------+-------------+----------+
| BAAI/bge-reranker-base                 | normal    | None      | xlm-roberta-base    | Chinese and English | Lightweight reranker model, easy to deploy, with fast inferenc         | BGEReranker | reranker |
.....
+----------------------------------------+-----------+-----------+---------------------+---------------------+------------------------------------------------------------------------+-------------+----------+
```

## 下载模型
```
$ python -m zerollama.tasks.reranker.cli pull BAAI/bge-reranker-v2-m3
```


## demo 服务端启动
```
$ python -m zerollama.tasks.reranker.inference_engine.server
use inference backend:
zerollama.models.baai.backend.reranker:BGEReranker
ZeroNameServer: InMemoryNameServer running! port: 9527
ZeroRetrieverInferenceEngine:  BAAI/bge-reranker-v2-m3 is running! port: 50939
```

## demo 客户端运行
```
$ python -m zerollama.tasks.reranker.inference_engine.client
================================================================================
Wait BAAI/bge-reranker-v2-m3 available
[ServerInfo(name='BAAI/bge-reranker-v2-m3', host='localhost', port=51312, protocol='reranker')]
================================================================================
ZeroRetrieverInferenceEngine support_methods
state='ok' msg={'support_methods': ['inference', 'info', 'support_methods']}
state='ok' msg={'name': 'BAAI/bge-reranker-v2-m3', 'type': 'normal', 'layerwise': None, 'Base model': 'bge-m3', 'Language': 'Multilingual', 'feature': 'Lightweight reranker model, possesses strong multilingual capabilities, easy to deploy, with fast inference.', 'family': 'BGEReranker', 'protocol': 'reranker'}
================================================================================
[0.00027803096387751553, 0.9948403768236574]
```